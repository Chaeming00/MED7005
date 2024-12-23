import timm
import torch
from torch import nn
from torchvision.models import DenseNet121_Weights
import torchvision.models as models
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x
class ImageEncoder_DenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x

class ImageEncoder_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class ImageEncoder_VIT(nn.Module):
    def __init__(
            self, model_name="vit_base_patch32_224", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

 
class STco(nn.Module):
    def __init__(self, max_x, max_y, temperature, image_embedding, spot_embedding, projection_dim, dropout=0.): #이부분 max_x,y 추가가
        super().__init__()
        self.x_embed = nn.Embedding(max_x, spot_embedding) #숫자 수정
        self.y_embed = nn.Embedding(max_y, spot_embedding) #숫자 수정
        self.image_ecode = ImageEncoder_DenseNet121()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim, dropout=dropout)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim, dropout=dropout)

        self.temperature = temperature


    def forward(self, batch):
        
        image_features = self.image_ecode(batch["image"])
        spot_features = batch["expression"]
        image_embeddings = self.image_projection(image_features)
        x = batch["position"][:, 0].long()
        y = batch["position"][:, 1].long()
        centers_x = self.x_embed(x)
        centers_y = self.y_embed(y)

        spot_features = spot_features + centers_x + centers_y

        spot_embeddings = self.spot_projection(spot_features)
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        targets = torch.eye(logits.shape[0], logits.shape[1]).cuda()
        spots_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()


