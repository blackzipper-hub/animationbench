import torch
# from torchcodec.decoders import VideoDecoder
import numpy as np
from transformers import AutoVideoProcessor, AutoModel
from decord import VideoReader, cpu
import glob
import tqdm
import argparse
import os
import json

def get_video_features(model, video_path, processor, num_frames=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    if num_frames is not None:
        start = max(0, len(vr) - num_frames)
        frame_idx = np.arange(start, len(vr), 1)  # choosing some frames. here, you can define more complex sampling strategy
    else:
        frame_idx = np.arange(0, len(vr), 1)
    video = vr.get_batch(frame_idx).asnumpy()  # T x C x H x W
    video = processor(video, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**video)
    return outputs.last_hidden_state  # B x T x D

def calculate_novelty_score(model_path="facebook/vjepa2-vitl-fpc64-256", ref_video_folder=None, videos=None, num_frames=None):
    processor = AutoVideoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    ref_videos = os.listdir(ref_video_folder)
    ref_videos = [f for f in ref_videos if f.endswith('.mp4')]
    test_videos = os.listdir(videos)
    test_videos = [f for f in test_videos if f.endswith('.mp4')]
    # assert len(ref_videos) == len(test_videos)

    scores = []
    per_video = {}
    for i in tqdm.tqdm(ref_videos):
        if os.path.exists(os.path.join(videos, i)) is False:
            continue
        ref_feature = get_video_features(model, os.path.join(ref_video_folder, i), processor, num_frames)
        video_feature = get_video_features(model, os.path.join(videos, i), processor, num_frames)
        a_pooled = video_feature.mean(dim=1)  # [1,D]
        b_pooled = ref_feature.mean(dim=1)
        score = float(torch.cosine_similarity(a_pooled.view(1, -1), b_pooled.view(1, -1)).item())
        novelty = float(np.clip(1.0 - score, 0.0, 0.3))
        scores.append(novelty)
        per_video[i] = novelty

    if not scores:
        raise ValueError(
            f"No comparable .mp4 pairs found between video_folder={videos} and ref_video_folder={ref_video_folder}"
        )

    return float(np.mean(scores)), per_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/vjepa2-vitl-fpc64-256", help="path to the video feature extraction model")
    parser.add_argument("--ref_video_folder", type=str, required=True, help="path to the reference video")
    parser.add_argument("--video_folder", type=str, required=True, help="path to the folder containing videos to evaluate")
    parser.add_argument("--results_path", type=str, default=None, help="optional path to write a json result")
    args = parser.parse_args()

    novelty_score, per_video = calculate_novelty_score(
        model_path=args.model_path,
        ref_video_folder=args.ref_video_folder,
        videos=args.video_folder,
        num_frames=100,
    )
    print(f"Novelty Score: {novelty_score}")

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
        with open(args.results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_path": args.model_path,
                    "ref_video_folder": args.ref_video_folder,
                    "video_folder": args.video_folder,
                    "num_frames": 100,
                    "novelty_score": novelty_score,
                    "per_video": per_video,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )