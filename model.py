"""
Two-stage model architecture for TB / DR-TB prediction.

Stage 1 (TBImageClassifier): CXR image -> TB vs Normal.
    Trained on TB_Chest_Radiography_Database, where the image really is the
    ground truth signal (label comes directly from the image's source
    folder), each image used exactly once.

Stage 2 (DRTBRiskModel): clinical + genomic features -> DR-TB risk.
    Trained on per-patient clinical/genomic records. No image input, because
    no dataset in this project actually pairs a real chest X-ray with that
    same patient's real drug-susceptibility result -- chest X-ray appearance
    is not a validated indicator of resistance status in the first place.

These two stages replace the previous single MultimodalFusionModel, which
fused CXR + clinical + genomic features into one DR-TB prediction. That model
was trained on a merged dataset where each of the 4,200 real X-ray images was
duplicated up to 53 times and paired with independently-generated synthetic
patient records, so the image had no real relationship to the resistance
label it was being trained to help predict.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for modality fusion. Works over any
    number of stacked modality embeddings (used with 2 modalities in
    DRTBRiskModel)."""

    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        output = self.out_proj(attn_output)
        return output, attn_weights.mean(dim=1)  # Average over heads


class TBImageClassifier(nn.Module):
    """Stage 1: chest X-ray -> TB vs Normal. Image-only.

    Compact CNN trained from scratch on the 4,200 real, uniquely-labeled
    images in TB_Chest_Radiography_Database. The original design used an
    ImageNet-pretrained EfficientNet-B4 backbone; retraining here could not
    fetch those pretrained weights because torchvision's weight host
    (download.pytorch.org) and the common mirror (huggingface.co) are both
    blocked by this environment's network egress policy. In an environment
    with normal network access, swap this for:
        from torchvision.models import EfficientNet_B4_Weights
        models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    with its classifier head replaced, which should outperform a from-scratch
    CNN on a dataset this size.
    """

    def __init__(self, num_classes=1):
        super(TBImageClassifier, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, cxr_image):
        x = self.features(cxr_image)
        x = self.pool(x)
        return self.classifier(x)


class DRTBRiskModel(nn.Module):
    """Stage 2: clinical + genomic features -> DR-TB risk. No image input."""

    def __init__(self, num_clinical_features, num_genomic_features, num_classes=1):
        super(DRTBRiskModel, self).__init__()

        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        clinical_features = 32

        self.genomic_encoder = nn.Sequential(
            nn.Linear(num_genomic_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )
        genomic_features = 16

        self.clinical_norm = nn.LayerNorm(clinical_features)
        self.genomic_norm = nn.LayerNorm(genomic_features)

        self.modality_dim = 64
        self.clinical_proj = nn.Linear(clinical_features, self.modality_dim)
        self.genomic_proj = nn.Linear(genomic_features, self.modality_dim)

        self.attention = MultiHeadAttention(embed_dim=self.modality_dim, num_heads=2)

        total_features = self.modality_dim * 2  # 2 modalities after attention
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, clinical_features, genomic_features):
        clinical_encoded = self.clinical_norm(self.clinical_encoder(clinical_features))  # (batch, 32)
        genomic_encoded = self.genomic_norm(self.genomic_encoder(genomic_features))  # (batch, 16)

        clinical_proj = self.clinical_proj(clinical_encoded)  # (batch, modality_dim)
        genomic_proj = self.genomic_proj(genomic_encoded)  # (batch, modality_dim)

        modalities = torch.stack([clinical_proj, genomic_proj], dim=1)  # (batch, 2, modality_dim)
        attended, attn_weights = self.attention(modalities)  # attn_weights: (batch, 2, 2)

        attended_features = attended.view(attended.size(0), -1)  # (batch, modality_dim*2)
        x = self.fusion_layer(attended_features)
        output = self.classifier(x)

        # Contribution of each modality to itself, averaged over the query axis
        modality_weights = attn_weights.mean(dim=1)  # (batch, 2) -> [clinical, genomic]

        return output, modality_weights
