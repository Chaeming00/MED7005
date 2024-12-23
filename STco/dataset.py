import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import numpy as np
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
import pandas as pd
import scprep as scp
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms.functional as TF
import random
import json
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from anndata import read_h5ad
import h5py

def calculate_max_coords(data_root, cancer_type):
    import h5py
    import os
    max_x, max_y = 0, 0
    patches_dir = os.path.join(data_root, cancer_type, 'patches')
    print(f"Scanning directory: {patches_dir}")
    for h5_file in os.listdir(patches_dir):
        if h5_file.endswith('.h5'):
            print(f"Processing file: {h5_file}")
            h5_path = os.path.join(patches_dir, h5_file)
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]
                max_x = max(max_x, coords[:, 0].max())
                max_y = max(max_y, coords[:, 1].max())
    print(f"Max X: {max_x}, Max Y: {max_y}")
    return max_x, max_y


class HESTDataset(Dataset):
    def __init__(self, data_root, cancer_type, split, gene_list_path, train=True, normalize=True, fold=0):
        """
        HEST 데이터셋을 PyTorch Dataset으로 변환
        :param data_root: 데이터 루트 디렉토리
        :param cancer_type: 암종 (예: 'LUNG', 'CCRCC' 등)
        :param split: 'train', 'test', 또는 'val'
        :param gene_list_path: 유전자 목록 파일 경로
        :param train: 학습 데이터 여부
        :param normalize: 유전자 발현 데이터 정규화 여부
        :param fold: 교차검증 fold 번호
        """
        super(HESTDataset, self).__init__()
 
        self.data_root = data_root
        self.cancer_type = cancer_type
        self.split = split
        self.train = train
        self.normalize = normalize
        self.fold = fold  # fold 번호 추가

        
        # 파일 경로 설정
        self.split_dir = os.path.join(self.data_root, self.cancer_type, 'splits')
        self.patches_dir = os.path.join(self.data_root, self.cancer_type, 'patches')
        self.adata_dir = os.path.join(self.data_root, self.cancer_type, 'adata')
        
        # split 파일 로드(각 line에 sample_id, patches_path, expr_path 등의 정보 있음)
        self.data = self.load_split_files(self.split_dir, split)

        # 유전자 목록 로드
        with open(gene_list_path, 'r') as f:
            gene_data = json.load(f)
            self.gene_list = gene_data['genes']

        # 데이터 증강 정의
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5, 0.5, 0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor() # (H, W, C) -> (C, H, W)?
        ]) if train else transforms.Compose([transforms.ToTensor()])
        
        # 첫번째 코드처럼 샘플별 데이터를 미리 로딩하고 전처리하기 위한 딕셔너리
        self.exp_dict = {}
        self.coords_dict = {}
        self.img_dict = {}
        self.sample_indices = []  # (sample_id, spot_index) 형태로 전체 아이템 인덱싱 관리

        # 데이터프레임의 각 행은 하나의 스팟을 의미한다고 가정하기보다는,
        # 여기서는 first code와 같이 "하나의 sample_id에 대해 여러 스팟"을 가지므로
        # sample_id 별로 데이터를 모아야 함.
        # 우선 data에서 unique한 sample_id 목록을 얻는다.
        unique_samples = self.data['sample_id'].unique()

        for sid in unique_samples:
            # 해당 sample_id에 대한 모든 행(spot) 추출
            sample_rows = self.data[self.data['sample_id'] == sid]

            # h5파일에서 이미지 패치 및 coords 로드
            h5_path = os.path.join(self.data_root, self.cancer_type, sample_rows.iloc[0]['patches_path'])
            with h5py.File(h5_path, 'r') as f:
                img_data = f['img'][:]      # (N_spots, H, W, C)
                coords = f['coords'][:]     # (N_spots, 2)
                # barcode = f['barcode'][:] # 필요시

            # expr_path에서 유전자 데이터 로드
            expr_path = os.path.join(self.data_root, self.cancer_type, sample_rows.iloc[0]['expr_path'])
            adata = read_h5ad(expr_path)
            genes = adata[:, self.gene_list].X.toarray()  # (N_spots, N_genes)
            
            # 여기서 한번에 정규화/로그 변환
            if self.normalize:
                genes = scp.normalize.library_size_normalize(genes)
                genes = scp.transform.log(genes)
                

            # 딕셔너리에 저장
            self.exp_dict[sid] = genes  # (N_spots, N_genes)
            self.coords_dict[sid] = coords  # (N_spots, 2)
            self.img_dict[sid] = img_data   # (N_spots, H, W, C)

            # sample_id별 spot 인덱스를 전역 인덱스로 매핑
            spot_count = genes.shape[0]
            for i in range(spot_count):
                self.sample_indices.append((sid, i))

    def load_split_files(self, split_dir, split_prefix):
        """
        주어진 split_prefix에 해당하는 모든 CSV 파일을 로드하여 병합
        :param split_dir: splits 폴더 경로
        :param split_prefix: 'train' 또는 'test'와 같은 prefix
        :return: 병합된 DataFrame
        """
        file_name = f"{split_prefix}_{self.fold}.csv"  # 예: train_0.csv
        file_path = os.path.join(split_dir, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        return pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # 샘플 정보
        sid, spot_idx = self.sample_indices[idx]
        
        # 이미지 선택 및 변환
        patch = self.img_dict[sid][spot_idx] 
        # numpy.ndarray를 PIL.Image로 변환
        patch = Image.fromarray(patch)
        patch = self.transforms(patch)
        
        loc = self.coords_dict[sid][spot_idx]  # (2,)
        genes = self.exp_dict[sid][spot_idx]   # (N_genes,)
        genes = torch.tensor(genes, dtype=torch.float32)
        loc = torch.tensor(loc, dtype=torch.float32)

         # 최종 데이터 반환
        return {
            "image": patch,               # 이미지 데이터
            "position": loc,  # 좌표 데이터
            "expression": genes,          # 유전자 발현 데이터
            "sample_id": sid  # 샘플 ID
        }