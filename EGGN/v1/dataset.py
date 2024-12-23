from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
import pandas as pd
import tifffile
from PIL import Image
from scanpy import read_visium
import scanpy as sc
import h5py
import json

class TxPDataset(Dataset):
    def __init__(self, root_dir, cancer_type, transform, args, train=None):
        """
        root_dir: 상위 디렉토리 경로 (예: '/path/to/10xgenomics')
        cancer_type: 암종 이름 (예: 'CCRCC', 'COAD')
        """
        print(f"Initializing TxPDataset with root_dir={root_dir}, cancer_type={cancer_type}")

        self.root_dir = root_dir
        self.cancer_type = cancer_type
        self.args = args
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(root_dir, cancer_type)
        
        # Load the selected genes from var_50genes.json
        var_50genes_path = os.path.join(self.data_dir, 'var_50genes.json')
        with open(var_50genes_path, 'r') as f:
            self.gene_names = json.load(f)["genes"]  # 유전자 이름 리스트를 불러옵니다
        
        # 데이터 로드
        self.data = self.load_raw()

        # 데이터의 최소값 및 최대값 계산
        self.min, self.max = self.calculate_min_max()

    def load_raw(self):
        """
        Load raw data for the specified cancer type.
        """
        data = []
        splits_dir = os.path.join(self.data_dir, 'splits')
        patches_dir = os.path.join(self.data_dir, 'patches')
        adata_dir = os.path.join(self.data_dir, 'adata')

        
        # Load CSV splits (e.g., train_0.csv, test_0.csv)
        if self.train:
            split_file = f"train_{self.args.fold}.csv"
        else:
            split_file = f"test_{self.args.fold}.csv"

        csv_path = os.path.join(splits_dir, split_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Split file not found: {csv_path}")

        split_data = pd.read_csv(csv_path)

        for _, row in split_data.iterrows():
            sample_id = row['sample_id']
            patch_path = os.path.join(self.data_dir, row['patches_path'])
            adata_path = os.path.join(self.data_dir, row['expr_path'])
           
            if not os.path.exists(patch_path):
                raise FileNotFoundError(f"Patch file not found: {patch_path}")
            if not os.path.exists(adata_path):
                raise FileNotFoundError(f"Adata file not found: {adata_path}")


            print(f"Loading h5ad file from: {adata_path}")
            print(f"Loading patch file from: {patch_path}")
            # Load h5ad data
            h5ad_data = sc.read_h5ad(adata_path)

            # Add sample_id to AnnData.uns
            h5ad_data.uns["sample_id"] = sample_id


            # Filter gene expression matrix to selected genes
            if hasattr(h5ad_data.X, 'todense'):
                counts = pd.DataFrame(h5ad_data.X.todense(), columns=h5ad_data.var_names, index=h5ad_data.obs_names)
            else:
                counts = pd.DataFrame(h5ad_data.X, columns=h5ad_data.var_names, index=h5ad_data.obs_names)


            # Filter data by predefined genes
            counts = counts.reindex(columns=self.gene_names, fill_value=0)  # Filter by gene list

            # Save filtered counts in `uns`
            h5ad_data.uns["filtered_counts"] = counts.values

    

            # Load patch data
            with h5py.File(patch_path, 'r') as h5_file:
                if 'img' in h5_file and 'coords' in h5_file:
                    img = h5_file['img'][()]
                    coords = h5_file['coords'][()]

                    # 좌표 정규화 (패치 단위 좌표로 변환)
                    patch_size = img.shape[1:3]  # (224, 224)

                    print(f"Image shape: {img.shape}, Coords shape: {coords.shape}")

    
                else:
                    raise KeyError(f"'img' or 'coords' dataset not found in {patch_path}")

                # Append valid data
                data.append([img, h5ad_data])

        return data

    def calculate_min_max(self):
        """
        Calculate the minimum and maximum values of the dataset.
        """
        all_data = []
        for _, h5 in self.data:
            counts = h5.uns["filtered_counts"]  # 이미 필터링된 데이터를 사용
            all_data.append(counts)

        all_data = np.concatenate(all_data, axis=0)
        return np.min(all_data), np.max(all_data)




    def meta_info(self):
        """
        Collect meta information for gene names and normalization.
        """
        from tqdm import tqdm
        gene_names = set()
        for _, h5 in tqdm(self.data):
            # h5.X가 sparse라면 todense() 호출, 아니라면 바로 DataFrame 변환
            if hasattr(h5.X, 'todense'):
                counts = pd.DataFrame(h5.X.todense(), columns=h5.var_names, index=h5.obs_names)
            else:
                counts = pd.DataFrame(h5.X, columns=h5.var_names, index=h5.obs_names)

            gene_names = gene_names.union(set(counts.columns.values))

        gene_names = list(gene_names)
        gene_names.sort()
        self.gene_names = gene_names


    def __getitem__(self, index):
        img_batch, h5 = self.data[index]
        counts = h5.uns["filtered_counts"]

        # Use coordinates as 'coords'
        if 'pxl_row_in_fullres' in h5.obs.columns and 'pxl_col_in_fullres' in h5.obs.columns:
            coords = h5.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
        elif 'array_row' in h5.obs.columns and 'array_col' in h5.obs.columns:
            coords = h5.obs[['array_row', 'array_col']].values
        else:
            raise KeyError("No suitable columns for coordinates found in h5.obs")

        imgs = []  # To store all processed patches
        for i in range(img_batch.shape[0]):  # Iterate over all patches
            img = img_batch[i]  # Select each patch
            if self.transform:
                img = Image.fromarray(img)
                img = self.transform(img)
            else:
                if len(img.shape) == 2:  # Grayscale image case
                    img = np.expand_dims(img, axis=-1)
                img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1) / 255.0
            imgs.append(img)

        imgs = torch.stack(imgs)  # Shape: [N, C, H, W]

        # Return the necessary keys including 'coords'
        # __getitem__ 메서드 수정
        counts = counts.astype(np.float32)  # Convert to float32
        p_feature = torch.mean(torch.tensor(counts, dtype=torch.float32), dim=0)

        return {
            "img": imgs,
            "coords": torch.tensor(coords, dtype=torch.float),
            "pos": torch.tensor(coords, dtype=torch.float),  # pos 추가
            "count": torch.tensor(counts, dtype=torch.float32),
            "p_feature": p_feature,  # Add processed feature
            "op_feature": p_feature.clone(),  # Placeholder for op_feature
            "op_count": torch.tensor(counts, dtype=torch.float32)  # Placeholder for op_count
        }



    def __len__(self):
        return len(self.data)


