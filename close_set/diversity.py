import sys
from pathlib import Path

# Allow direct execution via an absolute path by exposing the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.models import vgg19
from torch import nn
from vbench2.utils import get_frames
import os
import json
from tqdm import tqdm
import argparse

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(pretrained=True).features.eval()

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {0, 5, 10, 19, 28, 30}:
                features.append(x.detach().cpu())
        return features

def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.shape
    features = tensor.view(batch_size, channels, -1)
    gram = torch.bmm(features, features.transpose(1,2))  
    gram = gram / (channels * height * width)
    return gram

def content_loss(content, target_content):
    return torch.mean(torch.abs(content - target_content))

def style_loss(style, target_style):
    gram_style = gram_matrix(style)
    gram_target_style = gram_matrix(target_style)
    return torch.mean(torch.abs(gram_style - gram_target_style))

def evaluate(style_features, content_features):
    content_diversity = 0
    style_diversity = 0
    len_seed = len(content_features)
    for i in range(len_seed):
        for j in range(i+1, len_seed):
            content_diversity += content_loss(content_features[i], content_features[j])
            for k in range(5):
                style_diversity += style_loss(style_features[i][k], style_features[j][k])
    content_diversity/=(0.5*len_seed*(len_seed-1))
    style_diversity/=(2.5*len_seed*(len_seed-1))
    diversity=(content_diversity+1000*style_diversity)/2
    return content_diversity, 1000*style_diversity, diversity / 17.712 # Empirical maximum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, help='Path to the metadata JSON file containing video-question mappings')
    parser.add_argument('--video_folder', type=str, help='Path to the folder containing videos to evaluate')
    parser.add_argument('--results_path', type=str, help='Path to the output JSON file to save detailed results', default='diversity_results.json')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG().to(device)
    with open(args.metadata, 'r') as f:
        data = json.load(f)

    final_score=0
    processed_json=[]
    details = {}
    for k,v in tqdm(data.items()):
        style_features=[]
        content_features=[]
        for _ in range(5):
            video_path = os.path.join(args.video_folder, f"{k}_{_}.mp4")
            # print(video_path)
            frames=get_frames(video_path)
            frames=torch.cat(frames, dim=0)
            frames=frames.to(device)
            with torch.no_grad():
                features = model(frames)
            style=features[:5] 
            content=features[5]
            style_features.append(style)
            content_features.append(content)
            del style, content, frames
            torch.cuda.empty_cache()

        content_diversity, style_diversity, diversity=evaluate(style_features, content_features)
        diversity = torch.clamp(diversity, min=0, max=1)
        final_score+=diversity
        details[k] = {
            "overall_diversity": diversity.item()
        }
    
    with open(args.results_path, 'w') as f:
        json.dump(details, f, indent=4)
    print(final_score/len(data))
