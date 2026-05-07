import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from math import ceil
from tqdm import tqdm
import json
import os
from tqdm import tqdm
import json
import argparse
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return scene_list

def transform(vector):
    x = np.mean([item[0] for item in vector])
    y = np.mean([item[1] for item in vector])
    return [x, y]

def transform_class(vector, min_reso, factor=0.005): # 768*0.05
    scale = min_reso * factor
    x, y = vector
    direction = []
    if x > scale:
        direction.append("right")
    elif x < -scale:
        direction.append("left")
    if y > scale:
        direction.append("down")
    elif y < -scale:
        direction.append("up")
    return direction if direction else ["static"]

def transform_class360(vector, min_reso, factor=0.008): # 768*0.05
    scale = min_reso * factor
    up, down, y = vector
    if abs(y)<scale:
        if up * down<0 and up>scale:
            return "orbits"  #orbits_counterclockwise
        elif up*down<0 and up<-scale:
            return "orbits"   #orbits_clockwise
        else:
            return None

class CameraPredict:
    def __init__(self, device, submodules_list):
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        try:
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        except:
            # workaround for CERTIFICATE_VERIFY_FAILED (see: https://github.com/pytorch/pytorch/issues/33288#issuecomment-954160699)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)

    def transform360(self, vector):
        up=[]
        down=[]
        for item in vector:
            if item[2]>self.scale/2:
                down.append(item[0])
            else:
                up.append(item[0])
        y = np.mean([item[1] for item in vector])
        if len(up)>0:
            mean_up=sum(up)/len(up)
        else:
            mean_up=0
        if len(down)>0:
            mean_down=sum(down)/len(down)
        else:
           mean_down=0
        return [mean_up, mean_down, y]

    def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos"):
        b,_,_,h,w=video.shape
        self.scale=min(h,w)
        self.height=h
        self.width=w
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        return pred_tracks[0].detach().cpu().numpy()
    
    def get_edge_point(self, track):
        middle = self.grid_size // 2
        number = self.number_points / 2.0
        start = ceil(middle-number)
        end = ceil(middle+number)
        idx=0
        top = [list(track[idx, i, :]) for i in range(start, end)]
        down = [list(track[self.grid_size-idx-1, i, :]) for i in range(start, end)]
        left = [list(track[i, idx, :]) for i in range(start, end)]
        right = [list(track[i, self.grid_size-idx-1, :]) for i in range(start, end)]
        return top, down, left, right

    def get_edge_point_360(self, track):
        middle = self.grid_size // 2
        number = 2
        lists=[0,1,self.grid_size-2,self.grid_size-1]
        idx=2
        res=[]
        for i in lists:
            if track[i, idx, 0]<0 or track[i, idx, 1]<0:
                res.append(None)
            else:
                res.append(list(track[i, idx, :]))
        return res
    
    def get_edge_direction_360(self, tracks):
        # Disable orbit detection and keep the rest of camera motion labels unchanged.
        return []
    
    def check_valid(self, point):
        if point is not None:
            if point[0]>0 and point[0]<self.width and point[1]>0 and point[1]<self.height:
                return True
            else:
                return False
        else:
            return False
        
    def get_edge_direction(self, track1, track2):
        edge_points1 = self.get_edge_point(track1)
        edge_points2 = self.get_edge_point(track2)
        vector_results = []
        for points1, points2 in zip(edge_points1, edge_points2):
            vectors = [[end[0]-start[0], end[1]-start[1], start[1]] for start, end in zip(points1, points2)]
            vector_results.append(vectors)
        vector_results_pan = list(map(transform, vector_results)) 
        class_results = [transform_class(vector, min_reso=self.scale) for vector in vector_results_pan]
        return class_results

    def classify_top_down(self, top, down):
        results = []
        classes = [f"{item_t}_{item_d}" for item_t in top for item_d in down]
        results_mapping = {
            "left_left": "camera pan right.",
            "right_right": "camera pan left.",
            "down_down": "camera tilts up.",
            "up_up": "camera tilts down.",
            "up_down": "camera zooms in.",
            "down_up": "camera zooms out.",
            "static_static": "static shot (camera fixed)."
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else [None]
    
    def classify_left_right(self, left, right):
        results = []
        classes = [f"{item_l}_{item_r}" for item_l in left for item_r in right]
        results_mapping = {
            "left_left": "camera pan right.",
            "right_right": "camera pan left.",
            "down_down": "camera tilts up.",
            "up_up": "camera tilts down.",
            "left_right": "camera zooms in.",
            "right_left": "camera zooms out.",
            "static_static": "static shot (camera fixed)."
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else [None]


    def camera_classify(self, track1, track2, tracks):
        top, down, left, right = self.get_edge_direction(track1, track2)
        r360_results = self.get_edge_direction_360(tracks)
        top_results = self.classify_top_down(top, down)
        left_results = self.classify_left_right(left, right)
        results = list(set(top_results + left_results + r360_results))
        if None in results and len(results)>1:
            results.remove(None) 
        if "camera tilts up." in results and "camera zooms in." in results:
            results.append("oblique")
        if "static shot (camera fixed)." in results and len(results)>1:
            results.remove("static shot (camera fixed).") 
        return results
    
    def predict(self, video, fps, end_frame):
        pred_track = self.infer(video, fps, end_frame)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        results = self.camera_classify(track1, track2, tracks)

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Path to the metadata JSON file containing video-question mappings')
    parser.add_argument('--video_folder', type=str, help='Path to the folder containing videos to evaluate')
    parser.add_argument('--results_path', type=str, help='Path to the output JSON file to save detailed results', default='camera_motion_results.json')
    args = parser.parse_args()

    submodules_dict = {
        "repo":"facebookresearch/co-tracker",
        "model":"cotracker2"
    }

    device = "cuda:0"
    camera = CameraPredict(device, submodules_dict)
    details = {}
    score = 0

    with open(args.metadata, 'r') as f:
        data = json.load(f)
    for k,v in tqdm(data.items()):
        for _, prompt in enumerate(v):
            video_path = f'{args.video_folder}/{k}_{prompt.replace(" ", "_")}.mp4'
            print(f'Processing video: {video_path} with prompt: {prompt}')
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader))) 
            frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
            video = video.permute(0, 3, 1, 2)[None].float().to(device) # B T C H W
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            predict_results = camera.predict(video, fps, -1)
            if prompt in predict_results:
                score += 1
            print(f'Video: {video_path}, Prompt: {prompt}, Predicted: {predict_results}, Correct: {prompt in predict_results}')
            details[f"{k}_{prompt}"] = {
                "predicted": predict_results,
                "correct": prompt in predict_results
            }
    details['total_score'] = score/len(data)/7.0
    details['success'] = len(data)*7.0
    with open(args.results_path, 'w') as f:
        json.dump(details, f, indent=4)
    print(f'Final Score: {details["total_score"]}')
