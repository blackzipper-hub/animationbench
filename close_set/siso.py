import torch
import imageio.v3 as iio
import numpy as np
from PIL import Image
import argparse
import os
import json
import tqdm

def calculate_velocity_vector_stat_per_frame(pred_tracks, pred_visibility, stat="median"):
    """按帧聚合速度向量统计量（mean/median），返回 (dx, dy)。"""
    tracks = pred_tracks.detach().cpu().numpy()
    visibility = pred_visibility.detach().cpu().numpy()
    
    if visibility.ndim == 4:
        visibility = visibility[..., 0]
    
    displacements = tracks[:, 1:, :, :] - tracks[:, :-1, :, :]
    valid_mask = (visibility[:, :-1, :] > 0.5) & (visibility[:, 1:, :] > 0.5)
    disp_masked = np.where(valid_mask[..., None], displacements, np.nan)
    
    if stat == "mean":
        out = np.nanmean(disp_masked[0], axis=-2)
    else:
        out = np.nanmedian(disp_masked[0], axis=-2)
    
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def normalize_vectors(v, eps=1e-6):
    """逐行归一化向量，返回 (unit, norm)。"""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    unit = v / np.maximum(n, float(eps))
    return unit, n[..., 0]


def invert_binary_like_mask(mask):
    """反转二值/灰度mask。"""
    m = np.asarray(mask)
    maxv = float(np.nanmax(m)) if m.size else 255.0
    if maxv <= 1.0:
        return (1.0 - m).astype(m.dtype)
    return (maxv - m).astype(m.dtype)


def moving_average(x, window=9):
    """滑动平均平滑。"""
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or x.size == 0:
        return x.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x_pad, kernel, mode="valid")


def find_motion_bounds(speed):
    """找动作开始/结束位置。"""
    speed = np.asarray(speed, dtype=np.float32)
    if speed.size < 25:
        return {"start_idx": 0, "end_idx": max(0, speed.size - 1)}
    
    s = moving_average(speed, window=9)
    T = int(s.size)
    edge_trim = int(T * 0.02)
    left0 = max(0, edge_trim)
    right0 = min(T - 1, T - 1 - edge_trim)
    
    return {"start_idx": int(left0), "end_idx": int(right0)}


def detect_slow_in_slow_out_between_bounds(speed, start_idx, end_idx):
    """slow-in slow-out 打分（0~3）。"""
    speed = np.asarray(speed, dtype=np.float32)
    if speed.size < 10:
        return {"score": 0}
    
    s = moving_average(speed, window=9)
    T = int(s.size)
    start_idx = int(np.clip(start_idx, 0, T - 1))
    end_idx = int(np.clip(end_idx, 0, T - 1))
    
    if end_idx - start_idx < 10:
        return {"score": 0}
    
    s2 = s[start_idx:end_idx + 1]
    T2 = int(s2.size)
    seg = max(3, int(T2 * 0.2))
    if 2 * seg >= T2:
        seg = max(3, T2 // 3)
    
    start_seg = s2[:seg]
    end_seg = s2[-seg:]
    mid_seg = s2[seg:-seg] if (T2 - 2 * seg) > 0 else s2
    
    start_mean = float(np.mean(start_seg))
    mid_mean = float(np.mean(mid_seg))
    end_mean = float(np.mean(end_seg))
    
    delta_start = mid_mean - start_mean
    delta_end = mid_mean - end_mean
    rel_start = delta_start / max(1e-6, mid_mean)
    rel_end = delta_end / max(1e-6, mid_mean)
    
    cond_shape = (np.max(s2) / np.min(s2 + 1e-6)) >= 2.0
    cond_left_delta = (delta_start >= 0.15) and (rel_start >= 0.20)
    cond_right_delta = (delta_end >= 0.15) and (rel_end >= 0.20)
    
    score = int(cond_shape) + int(cond_left_delta) + int(cond_right_delta)
    return {"score": score}


def detect_accel_decel_not_too_fast(speed, start_idx, end_idx, total_frames):
    """评估加速/减速是否"不能太快"（0~2）。"""
    speed = np.asarray(speed, dtype=np.float32)
    if speed.size < 5:
        return {"score": 0}
    
    s = moving_average(speed, window=9)
    T = int(s.size)
    start_idx = int(np.clip(start_idx, 0, T - 1))
    end_idx = int(np.clip(end_idx, 0, T - 1))
    
    if end_idx - start_idx < 4:
        return {"score": 0}
    
    seg = s[start_idx:end_idx + 1]
    mean_val = float(np.mean(seg))
    tol = 0.20 * max(1e-6, mean_val)
    
    local_peak = int(np.argmax(seg))
    peak_val = float(seg[local_peak])
    peak_mask = (peak_val - seg) <= tol
    
    # 找峰值平台
    peak_l_local = local_peak
    while peak_l_local > 0 and peak_mask[peak_l_local - 1]:
        peak_l_local -= 1
    peak_r_local = local_peak
    while peak_r_local < len(seg) - 1 and peak_mask[peak_r_local + 1]:
        peak_r_local += 1
    
    peak_l = start_idx + peak_l_local
    peak_r = start_idx + peak_r_local
    
    # 左波谷
    left_seg = s[start_idx:peak_l + 1]
    left_local_min = int(np.argmin(left_seg))
    lv_r = start_idx + left_local_min
    
    # 右波谷
    right_seg = s[peak_r:end_idx + 1]
    right_local_min = int(np.argmin(right_seg))
    rv_l = peak_r + right_local_min
    
    left_dist = int(max(0, peak_l - lv_r))
    right_dist = int(max(0, rv_l - peak_r))
    
    min_frames = 0.05 * float(total_frames)
    accel_ok = float(left_dist) >= min_frames
    decel_ok = float(right_dist) >= min_frames
    
    score = int(accel_ok) + int(decel_ok)
    return {"score": int(score)}


def calculate_siso_score(video_path, mask_path, device='cuda'):
    """
    计算视频的 slow-in slow-out 总分。
    
    参数:
        video_path: 视频文件路径
        mask_path: mask图片路径
        device: 计算设备
    
    返回:
        总分 (0~5)
    """
    # 加载模型
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # 读取视频和mask
    grid_size = 100
    frames = iio.imread(video_path, plugin="FFMPEG")
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    
    segm_mask = np.array(
        Image.open(mask_path).convert("L").resize((video.shape[-1], video.shape[-2]))
    )
    bg_mask = invert_binary_like_mask(segm_mask)
    
    # 前景追踪
    pred_tracks, pred_visibility = cotracker(
        video,
        grid_size=grid_size,
        segm_mask=torch.from_numpy(segm_mask)[None, None],
    )
    torch.cuda.empty_cache()
    
    # 背景追踪
    bg_tracks, bg_visibility = cotracker(
        video,
        grid_size=grid_size // 4,
        segm_mask=torch.from_numpy(bg_mask)[None, None],
    )
    
    # 计算相对速度
    fg_vel_med = calculate_velocity_vector_stat_per_frame(pred_tracks, pred_visibility, stat="median")
    bg_vel_med = calculate_velocity_vector_stat_per_frame(bg_tracks, bg_visibility, stat="median")
    
    fg_dir_unit, fg_speed_med = normalize_vectors(fg_vel_med)
    bg_speed_proj_on_fg = np.sum(bg_vel_med * fg_dir_unit, axis=-1)
    
    rel_speed_signed = fg_speed_med - bg_speed_proj_on_fg
    rel_speed_per_frame = np.abs(rel_speed_signed).astype(np.float32)
    
    # 找动作边界
    bounds = find_motion_bounds(rel_speed_per_frame)
    start_idx, end_idx = bounds["start_idx"], bounds["end_idx"]
    
    # slow-in slow-out 打分
    siso = detect_slow_in_slow_out_between_bounds(rel_speed_per_frame, start_idx, end_idx)
    
    # 加速/减速打分
    accel_decel = detect_accel_decel_not_too_fast(
        rel_speed_per_frame, start_idx, end_idx, total_frames=int(video.shape[1])
    )
    
    total_score = int(siso["score"]) + int(accel_decel["score"])
    return total_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Path to the metadata JSON file containing video-question mappings')
    parser.add_argument('--video_folder', type=str, help='Path to the folder containing videos to evaluate')
    parser.add_argument('--mask_folder', type=str, help='Path to the folder containing mask images to evaluate')
    parser.add_argument('--results_path', type=str, help='Path to the output JSON file to save detailed results', default='siso_results.json')
    args = parser.parse_args()

    with open(args.metadata, 'r') as f:
        data = json.load(f)

    details = {}
    for k,v in tqdm.tqdm(data.items()):
        video_path = os.path.join(args.video_folder, f"{k}.mp4")
        mask_path = os.path.join(args.mask_folder, f"{k}_mask.png")
        score = calculate_siso_score(video_path, mask_path)
        print(f"Video: {k}.mp4, SISO Score: {score} / 5.0")
        details[k] = {
            "siso_score": score
        }

    with open(args.results_path, 'w') as f:
        json.dump(details, f, indent=4)