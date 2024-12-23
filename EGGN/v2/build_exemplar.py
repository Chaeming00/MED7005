import os
import torch
import tifffile
import heapq
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
from scanpy import read_visium
import scanpy as sc
import pandas as pd
from joblib import Parallel, delayed
import sys
sys.path.insert(0, "../")
from v1.dataset import TxPDataset  # TxPDataset 클래스 임포트
from v1.main import KFOLD
import torch.nn.functional as F

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--index_dir", required=True, type=str)
parser.add_argument("--save_dir", required=True, type=str)
parser.add_argument("--TORCH_HOME", required=True, type=str)
parser.add_argument("--data", required=True, type=str)
parser.add_argument("--fold", required=True, type=int)
parser.add_argument("--cancer_type", required=True, type=str)

args = parser.parse_args()

# Configuration
os.environ['TORCH_HOME'] = args.TORCH_HOME
cancer_type = args.cancer_type
save_dir = os.path.join(args.save_dir, cancer_type)
index_dir = args.index_dir
reference = KFOLD[args.fold][0]
num_cores = 12
batch_size = 64
window = 256

# Encoder Setup
encoder = torchvision.models.resnet18(pretrained=True)
features = encoder.fc.in_features
encoder = torch.nn.Sequential(*list(encoder.children())[:-1])
for p in encoder.parameters():
    p.requires_grad = False
encoder = encoder.cuda()
encoder.eval()

# Directory Setup
os.makedirs(os.path.join(save_dir, index_dir), exist_ok=True)

# Load Dataset
dataset = TxPDataset(
    root_dir=args.data,
    cancer_type=cancer_type,
    transform=None,
    args=args,
    train=True
)

if not dataset.data:
    print(f"Filtered dataset for {cancer_type} is empty. Skipping.")
    sys.exit(1)

print(f"Filtered dataset length for {cancer_type}: {len(dataset.data)}")

# Data Collection
data = []
for item in tqdm(dataset, desc="Collecting data"):
    img, coord = item['img'], item['coords']
    data.append([img, coord])

mapping = []
for i, (img, coord) in enumerate(data):
    num_patches = min(img.shape[0], len(coord))
    mapping.append(list(range(num_patches)))


# Generate Embeddings
def generate(data, mapping, save_dir, index_dir):
    def get_slide_gene(idx):
        img, coord = data[idx[0]]
        coord = coord[idx[1]]

        if len(img.shape) == 4:  # Batch of images
            if idx[1] >= img.shape[0]:
                return None, None
            img = img[idx[1]]

        if len(img.shape) == 3:
            if img.shape[-1] == 3:
                # [H, W, C] 형태 -> [C, H, W]로 변환
                img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1)
            elif img.shape[0] == 3:
                # 이미 [C, H, W] 형태
                img = torch.as_tensor(img, dtype=torch.float)
            else:
                raise ValueError(f"Unexpected 3D image shape: {img.shape}")
        elif len(img.shape) == 2:
            # Grayscale image [H, W] -> [C, H, W]
            img = torch.as_tensor(img[:, :, None], dtype=torch.float).permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # 이미지 정규화
        img = img / 255.0
        img = (img - 0.5) / 0.5


        return img, idx[1]


    def extract(imgs):
        imgs = torch.stack(imgs).cuda()
        return encoder(imgs).view(-1, features)

    cancer_dir = os.path.join(save_dir, index_dir)
    for i, k in tqdm(enumerate(mapping), total=len(mapping), desc="Generating embeddings"):
        batch_img, codes, embeddings = [], [], []
        for j in k:
            img, code = get_slide_gene([i, j])
            batch_img.append(img)
            codes.append(code)

            if len(batch_img) == batch_size:
                embeddings.append(extract(batch_img))
                batch_img = []

        if batch_img:
            embeddings.append(extract(batch_img))

        if embeddings:
            torch.save(torch.cat(embeddings), os.path.join(cancer_dir, f"{i}.pt"))


# Create Search Index
def create_search_index(mapping, save_dir, index_dir):
    class Queue:
        def __init__(self, max_size=2):
            self.max_size = max_size
            self.list = []

        def add(self, item):
            heapq.heappush(self.list, item)
            if len(self.list) > self.max_size:
                heapq.heappop(self.list)

    cancer_dir = os.path.join(save_dir, index_dir)
    embeddings = {i: torch.load(os.path.join(cancer_dir, f"{i}.pt")).cuda() for i in range(len(mapping))}

    for i in tqdm(range(len(mapping)), desc="Creating search index"):
        p = embeddings[i]
        Q = [Queue(max_size=128) for _ in range(p.size(0))]

        for op_i in range(len(mapping)):
            if op_i == i or op_i not in reference:
                continue
            op = embeddings[op_i]
            dist = torch.cdist(p.unsqueeze(0), op.unsqueeze(0), p=1).squeeze(0)

            knn = dist.topk(min(len(dist), 100), dim=1, largest=False)
            for f, (q_val, q_idx) in enumerate(zip(knn.values.cpu().numpy(), knn.indices.cpu().numpy())):
                for v, idx in zip(q_val, q_idx):
                    Q[f].add((-v, idx, op_i))

        np.save(os.path.join(save_dir, f"{i}.npy"), [q.list for q in Q])


# Main Execution
generate(data, mapping, save_dir, index_dir)
create_search_index(mapping, save_dir, index_dir)
print(f"Finished processing {cancer_type}. Results are saved to {save_dir}.")
