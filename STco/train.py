import argparse
import torch
import os
import torch.nn.functional as F

from model import STco
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AvgMeter, get_lr
from dataset import HESTDataset, calculate_max_coords
import glob
import json

ALL_CANCER_TYPES = ["CCRCC","COAD","HCC","IDC","LUNG", "LYMPH_IDC","PAAD", "PRAD", "READ","SKCM"]

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='') 
parser.add_argument('--max_epochs', type=int, default=100, help='')
parser.add_argument('--temperature', type=float, default=1., help='temperature')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--dim', type=int, default=50, help='spot_embedding dimension (# HVGs)')  # 171, 785 #숫자 수정
parser.add_argument('--image_embedding_dim', type=int, default=1024, help='image_embedding dimension')
parser.add_argument('--projection_dim', type=int, default=256, help='projection_dim ')
parser.add_argument('--dataset', type=str, default='HEST', help='dataset') #기본값 HEST로 수정
parser.add_argument('--cancer_types', nargs='+', default=ALL_CANCER_TYPES, help='List of cancer types to train on') #모든 암종 대해 학습되게

def get_fold_count(data_root, cancer_type):
    """
    특정 암종의 fold 수를 계산합니다.
    :param data_root: 데이터 루트 디렉토리
    :param cancer_type: 암종 (예: 'LUNG', 'CCRCC' 등)
    :return: fold 수
    """
    split_dir = os.path.join(data_root, cancer_type, 'splits')
    train_files = glob.glob(os.path.join(split_dir, 'train_*.csv'))
    return len(train_files)

def load_data(cancer_type, fold, args):

    # 데이터 루트 및 유전자 리스트 경로 설정
    data_root = "/home/guest3/projects/STco/HEST/bench_data"
    gene_list_path = f"/home/guest3/projects/STco/HEST/bench_data/{cancer_type}/var_50genes.json"
    

    if args.dataset == 'HEST':
        # 암종별 최대 x, y 계산
        
        max_x, max_y = calculate_max_coords(data_root, cancer_type)
        print(f"Max X: {max_x}, Max Y: {max_y} for Cancer Type: {cancer_type}")

        train_dataset = HESTDataset(
            data_root=data_root,
            cancer_type=cancer_type,
            split="train",
            gene_list_path=gene_list_path.format(cancer_type=cancer_type),
            train=True,
            fold=fold  # fold 번호 전달
        )
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = HESTDataset(
            data_root=data_root,
            cancer_type=cancer_type,
            split="test",
            gene_list_path=gene_list_path.format(cancer_type=cancer_type),
            train=False,
            fold=fold  # fold 번호 전달
        )

        return train_dataLoader, test_dataset, max_x, max_y

    if args.dataset == 'her2st':
        print(f'load dataset: {args.dataset}')
        train_dataset = HERDataset(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = HERDataset(train=False, fold=args.fold)
        return train_dataLoader, test_dataset
    elif args.dataset == 'cscc':
        print(f'load dataset: {args.dataset}')
        train_dataset = SKIN(train=True, fold=args.fold)
        train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = SKIN(train=False, fold=args.fold)
        return train_dataLoader, test_dataset


def train(model, train_dataLoader, optimizer, epoch):
    """
    Train the model for a single epoch.
    
    Args:
        model: PyTorch model to train.
        train_dataLoader: DataLoader for training data.
        optimizer: Optimizer for updating model weights.
        epoch: Current epoch number.

    Returns:
        Average loss for the epoch.
    """
    loss_meter = AvgMeter()
    tqdm_train = tqdm(train_dataLoader, total=len(train_dataLoader),desc=f"Training Epoch {epoch}")
    # Loop through batches
    for batch in tqdm_train:
        # Filter and move necessary keys to CUDA
        batch = {k: v.cuda() for k, v in batch.items() if
                 k == "image" or k == "expression" or k == "position"}
        # Forward pass
        loss = model(batch)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update loss meter
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        # Update tqdm progress bar
        tqdm_train.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer), epoch=epoch)
    return loss_meter.avg
    


def save_model(cancer_type,fold,model):
    save_path = f"./model_result/{cancer_type}/{fold}/best_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    args = parser.parse_args()
    data_root = "/home/guest3/projects/STco/HEST/bench_data"

    for cancer_type in args.cancer_types:  # 암종 루프
        print(f"Processing Cancer Type: {cancer_type}")
        
        # 암종별 fold 수 계산
        fold_count = get_fold_count(data_root, cancer_type)
        print(f"Number of folds for {cancer_type}: {fold_count}")

        for fold in range(fold_count):
            print(f"Processing {cancer_type}, Fold: {fold}")
            train_dataLoader, test_dataset,  max_x, max_y= load_data(cancer_type, fold, args)

            if train_dataLoader is None or len(train_dataLoader.dataset) == 0:
                raise ValueError(f"Training data not loaded correctly for {cancer_type}, Fold: {fold}. Check data files.")


            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = STco(spot_embedding=args.dim, temperature=args.temperature,
                                    image_embedding=args.image_embedding_dim, projection_dim=args.projection_dim, max_x=max_x + 1, max_y=max_y + 1).cuda()

            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-4, weight_decay=1e-3
            )
            
            # Fold별로 train_losses 초기화
            train_losses = []  # Fold별 초기화

            for epoch in range(args.max_epochs):
                model.train()
                avg_loss = train(model, train_dataLoader, optimizer, epoch)
                train_losses.append(avg_loss)


            save_model(cancer_type, fold, model) 
            print(f"Model saved for {cancer_type}, Fold: {fold}")
        

            # train_losses 저장 (중복 방지 포함)
            if not train_losses:
                print(f"No training losses to save for {cancer_type}, Fold: {fold}. Skipping...")
            else:
                output_path = f'train_losses_{cancer_type}_{fold}.json'
                if os.path.exists(output_path):
                    print(f"File {output_path} already exists. Overwriting...")
                # Save training losses
                with open(output_path, 'w') as f:
                    json.dump(train_losses, f)
                print(f"Training losses saved to {output_path}")

main()
