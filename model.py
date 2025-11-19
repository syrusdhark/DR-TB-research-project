"""
Multimodal Fusion Model for DR-TB Prediction
Extracted from the training notebook for use in the web interface.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for modality fusion."""
    
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
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output, attn_weights.mean(dim=1)  # Average over heads


class MultimodalFusionModel(nn.Module):
    """
    Enhanced multimodal fusion model with multi-head attention and residual connections.
    Combines CXR images, clinical metadata, and genomic features.
    """
    
    def __init__(self, num_clinical_features, num_genomic_features, num_classes=1):
        super(MultimodalFusionModel, self).__init__()
        
        # CXR Encoder: EfficientNet-B4
        # Use weights parameter instead of deprecated pretrained
        from torchvision.models import EfficientNet_B4_Weights
        self.cxr_encoder = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        cxr_features = 1792  # EfficientNet-B4 output features
        self.cxr_encoder.classifier = nn.Identity()
        
        # Enhanced Clinical Metadata Encoder with residual connections
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm for better stability
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
        
        # Enhanced Genomic Feature Encoder
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
        
        # Normalize features before fusion
        self.cxr_norm = nn.LayerNorm(cxr_features)
        self.clinical_norm = nn.LayerNorm(clinical_features)
        self.genomic_norm = nn.LayerNorm(genomic_features)
        
        # Multi-head attention for modality fusion
        # Project each modality to same dimension for attention
        self.modality_dim = 256
        self.cxr_proj = nn.Linear(cxr_features, self.modality_dim)
        self.clinical_proj = nn.Linear(clinical_features, self.modality_dim)
        self.genomic_proj = nn.Linear(genomic_features, self.modality_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim=self.modality_dim, num_heads=4)
        
        # Enhanced fusion with residual connections
        total_features = self.modality_dim * 3  # After attention, we have 3 modalities
        self.fusion_layer1 = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fusion_layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fusion_layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Residual connection for fusion layers
        self.fusion_residual1 = nn.Linear(total_features, 512)
        self.fusion_residual2 = nn.Linear(512, 256)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        # Simple attention for interpretability (backward compatibility)
        self.simple_attention = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, cxr_image, clinical_features, genomic_features):
        # Extract and normalize features
        cxr_features = self.cxr_norm(self.cxr_encoder(cxr_image))  # (batch_size, 1792)
        clinical_encoded = self.clinical_norm(self.clinical_encoder(clinical_features))  # (batch_size, 32)
        genomic_encoded = self.genomic_norm(self.genomic_encoder(genomic_features))  # (batch_size, 16)
        
        # Project to same dimension for attention
        cxr_proj = self.cxr_proj(cxr_features)  # (batch_size, 256)
        clinical_proj = self.clinical_proj(clinical_encoded)  # (batch_size, 256)
        genomic_proj = self.genomic_proj(genomic_encoded)  # (batch_size, 256)
        
        # Stack modalities for multi-head attention: (batch_size, 3, 256)
        modalities = torch.stack([cxr_proj, clinical_proj, genomic_proj], dim=1)
        
        # Apply multi-head attention
        attended_modalities, attn_weights = self.attention(modalities)  # (batch_size, 3, 256)
        
        # Flatten attended features
        attended_features = attended_modalities.view(attended_modalities.size(0), -1)  # (batch_size, 768)
        
        # Fusion with residual connections
        x = self.fusion_layer1(attended_features)
        x = x + self.fusion_residual1(attended_features)  # Residual connection
        
        x = self.fusion_layer2(x)
        x = x + self.fusion_residual2(self.fusion_layer1[0](attended_features))  # Residual connection
        
        x = self.fusion_layer3(x)
        
        # Final classification
        output = self.classifier(x)
        
        # Compute simple attention weights for interpretability (backward compatibility)
        simple_attn = self.simple_attention(attended_features)
        
        return output, simple_attn

