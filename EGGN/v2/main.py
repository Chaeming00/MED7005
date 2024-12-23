import argparse
import os
from model import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
from pytorch_lightning.plugins import DDPPlugin
import glob
cudnn.benchmark = True
import torch_geometric
import sys
sys.path.insert(0, "../")
from v1.main import KFOLD
from timm.layers import LayerNorm2d  # 이전 경로 대신 새 경로 사용

def load_dataset(graph_path, cancer_type, numk):
    graph_folder = os.path.join(graph_path, cancer_type, cancer_type, f"graphs_{numk}")
    print(f"Looking for .pt files in: {graph_folder}")

    all_files = glob.glob(f"{graph_folder}/*.pt")
    if not all_files:
        raise FileNotFoundError(f"No .pt files found in {graph_folder}")

    dataset = []
    for file in all_files:
        try:
            data = torch.load(file)
            print(f"Loaded {file}: type={type(data)}")

            # 'window' 노드 확인
            if 'window' not in data.node_types:
                print(f"'window' node type not found in {file}.")
                continue

            # 'x'와 'y' 속성 확인
            if data['window'].x is None or data['window'].y is None:
                print(f"'x' or 'y' attribute is None in 'window' of {file}.")
                continue

            print(f"File {file}: window.x shape = {data['window'].x.shape}, window.y shape = {data['window'].y.shape}")
            dataset.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    print(f"Total loaded dataset: {len(dataset)} samples.")
    return dataset



def main(args):
    # 암종별 결과 저장 폴더 설정
    results_dir = os.path.join(args.output, args.cancer_type)
    os.makedirs(results_dir, exist_ok=True)

    # 로그 파일 설정
    log_file = os.path.join(results_dir, "train_log.txt")
    def write_log(*strings):
        strings = [str(s) for s in strings]
        with open(log_file, "a") as f:
            f.write(" ".join(strings) + "\n")
    print = partial(write_log)
    print(f"Starting training for cancer type: {args.cancer_type}")

    # 데이터 로드
    train_dataset = load_dataset(args.graph_path, args.cancer_type, args.numk)

    # 데이터셋 비어있는 경우 처리
    if not train_dataset:
        print("Train dataset is empty. Exiting training process.")
        return

    train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=1, num_workers=4  # CPU 코어를 활용해 데이터 로드 속도 증가
    )

    # 모델 설정
    model = HeteroGNN(args.num_layers, args.mdim)
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir'])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay, results_dir)
    model = TrainerModel(config, model)

    # 학습 전 손실 확인 코드
    print("Checking initial loss...")
    for batch in train_loader:
        # 'window' 노드의 'x'와 'y' 접근
        try:
            pred = model(batch)  # 모델 예측 수행
            target = batch['window']['y']  # 'window' 노드의 'y'를 타겟으로 설정
            loss = model.criterion(pred, target)  # 손실 계산
            print(f"Initial Loss: {loss.item()}")  # 현재 손실 값 출력
        except Exception as e:
            print(f"Error calculating initial loss: {e}")
        break  # 첫 번째 배치만 확인하므로 break

    # 학습 시작
    trainer = pl.Trainer(
        max_epochs=args.epoch,
        gpus=args.gpus,
        strategy=DDPPlugin(find_unused_parameters=False),  # 성능 최적화를 위해 False로 변경
        enable_checkpointing=False,  # 최신 옵션으로 수정
        logger=False
    )

    trainer.fit(model, train_dataloaders=train_loader)
    checkpoint_path = os.path.join(results_dir, "model_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    print("Training completed!")

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--gpus", required=True, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--verbose_step", default=10, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--mdim", default=512, type=int)
    parser.add_argument("--output", default="results", type=str)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--graph_path", required=True, type=str)
    parser.add_argument("--cancer_type", required=True, type=str)
    parser.add_argument("--numk", default=6, type=int)

    args = parser.parse_args()
    main(args)
