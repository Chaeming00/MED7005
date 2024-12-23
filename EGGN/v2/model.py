import torch
import torch.nn as nn
import torch_geometric.nn as pyg
from geb import EGBBlock
from csra import CSRA
from heteroconv import HeteroConv

'''
code is based on https://pytorch-geometric.readthedocs.io/en/latest/

'''

    
class HeteroGNN(torch.nn.Module):
    def __init__(self, num_layers=4, mdim=512):
        super().__init__()

        hidden_channels = 512
        out_channels = 250
        input_channel = mdim

        # 수정: pretransform_win의 입력 크기를 1949에 맞춤
        self.pretransform_win = pyg.Linear(1949, hidden_channels, bias=False)  # 1949로 수정
        self.pretransform_exp = pyg.Linear(100, hidden_channels, bias=False)
        self.post_transform = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            pyg.Linear(hidden_channels, hidden_channels, bias=False),
            nn.LeakyReLU(0.2, True),
        )
        self.pretransform_ey = pyg.Linear(250, hidden_channels, bias=False)
        self.leaklyrelu = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('window', 'near', 'window'): pyg.SAGEConv(hidden_channels, hidden_channels),
                ('example', 'close', 'example'): pyg.SAGEConv(hidden_channels, hidden_channels),
                ('example', 'refer', 'window'): EGBBlock((hidden_channels, hidden_channels, hidden_channels), hidden_channels, hidden_channels, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)

        self.pool = CSRA(hidden_channels)
        self.lin = pyg.Linear(hidden_channels, 100)  # y 크기에 맞게 조정


    def forward(self, x_dict, edge_index_dict):
        # 데이터 크기 확인
        print(f"Shape of x_dict['window'] before pretransform: {x_dict['window'].shape}")

        # 크기 변환 로직 추가
        if x_dict["window"].dim() != 2 or x_dict["window"].shape[1] != 1949:
            print(f"Adjusting x_dict['window'] shape from {x_dict['window'].shape}")
            x_dict["window"] = x_dict["window"].unsqueeze(1) if x_dict["window"].dim() == 1 else x_dict["window"]
            x_dict["window"] = torch.nn.functional.interpolate(x_dict["window"].unsqueeze(0), size=1949).squeeze(0)

        print(f"Shape of x_dict['window'] after resizing: {x_dict['window'].shape}")

        x_dict["example"] = self.post_transform(self.pretransform_exp(x_dict["example"]))
        x_dict['window'] = self.post_transform(self.pretransform_win(x_dict['window']))
        x_dict["example_y"] = self.pretransform_ey(x_dict["example"][:, -250:])

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.leaklyrelu(x) for key, x in x_dict.items()}
        return self.lin(self.pool(x_dict, edge_index_dict))



 