"""
Multimodal Age Group Prediction Model
Inspired by LSM-2 (Large Sensor Model) architecture with Adaptive and Inherited Masking (AIM)
for handling incomplete wearable sensor data.

Architecture:
- Two separate transformer encoders for AortaP and BrachP modalities
- Attention masking to handle missing data (inspired by AIM)
- Cross-modal fusion layer
- Classification head for 6 age groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=336, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder with attention masking for missing data (AIM-inspired)"""
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - True for missing values (to be masked)
        """
        # Create attention mask: True means ignore (mask out)
        if mask is not None:
            # Expand mask for attention: [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            # For each position, mask out all positions where mask is True
            attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, seq_len, seq_len]
        else:
            attn_mask = None
        
        # Apply transformer
        output = self.transformer(x, src_key_padding_mask=mask)
        return output


class ModalityEncoder(nn.Module):
    """Encoder for a single modality (AortaP or BrachP)"""
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_len=336):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        
        # Transformer encoder with attention masking
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len] - time series data
            mask: [batch_size, seq_len] - True for missing values
        """
        # Project to d_model
        x = self.input_projection(x.unsqueeze(-1))  # [B, seq_len, 1] -> [B, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer with attention masking
        x = self.transformer(x, mask=mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class CrossModalFusion(nn.Module):
    """Cross-modal fusion layer to combine AortaP and BrachP features"""
    def __init__(self, d_model=128, fusion_dim=256, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, aorta_features, brach_features):
        """
        Args:
            aorta_features: [batch_size, seq_len, d_model]
            brach_features: [batch_size, seq_len, d_model]
        """
        # Concatenate features
        fused = torch.cat([aorta_features, brach_features], dim=-1)  # [B, seq_len, 2*d_model]
        
        # Apply fusion
        fused = self.fusion(fused)  # [B, seq_len, fusion_dim]
        
        return fused


class AgeGroupPredictor(nn.Module):
    """Multimodal Age Group Prediction Model"""
    def __init__(self, 
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 fusion_dim=256,
                 num_classes=6,
                 dropout=0.1,
                 max_len=336,
                 pooling='mean'):
        super().__init__()
        
        # Two modality encoders
        self.aorta_encoder = ModalityEncoder(
            input_dim=1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
        
        self.brach_encoder = ModalityEncoder(
            input_dim=1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            d_model=d_model,
            fusion_dim=fusion_dim,
            dropout=dropout
        )
        
        # Pooling strategy
        self.pooling = pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, aorta_data, brach_data, aorta_mask=None, brach_mask=None):
        """
        Args:
            aorta_data: [batch_size, seq_len] - AortaP time series
            brach_data: [batch_size, seq_len] - BrachP time series
            aorta_mask: [batch_size, seq_len] - True for missing AortaP values
            brach_mask: [batch_size, seq_len] - True for missing BrachP values
        Returns:
            logits: [batch_size, num_classes]
        """
        # Encode each modality
        aorta_features = self.aorta_encoder(aorta_data, mask=aorta_mask)  # [B, seq_len, d_model]
        brach_features = self.brach_encoder(brach_data, mask=brach_mask)  # [B, seq_len, d_model]
        
        # Fuse modalities
        fused_features = self.fusion(aorta_features, brach_features)  # [B, seq_len, fusion_dim]
        
        # Pool over sequence length
        if self.pooling == 'mean':
            # Mean pooling with attention to missing values
            if aorta_mask is not None or brach_mask is not None:
                # Create combined mask: True if either modality is missing
                combined_mask = torch.zeros_like(aorta_data, dtype=torch.bool)
                if aorta_mask is not None:
                    combined_mask = combined_mask | aorta_mask
                if brach_mask is not None:
                    combined_mask = combined_mask | brach_mask
                # Invert mask for valid positions
                valid_mask = ~combined_mask  # [B, seq_len]
                # Compute mean only over valid positions
                pooled = (fused_features * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = fused_features.mean(dim=1)
        elif self.pooling == 'max':
            pooled = fused_features.max(dim=1)[0]
        else:  # cls token or last token
            pooled = fused_features[:, 0, :]  # Use first token
        
        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return logits


def create_model(config=None):
    """Create model with default or custom configuration"""
    if config is None:
        config = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512,
            'fusion_dim': 256,
            'num_classes': 6,
            'dropout': 0.1,
            'max_len': 336,
            'pooling': 'mean'
        }
    
    model = AgeGroupPredictor(**config)
    return model


if __name__ == "__main__":
    # Test model
    batch_size = 4
    seq_len = 336
    
    # Create model
    model = create_model()
    
    # Create dummy data
    aorta_data = torch.randn(batch_size, seq_len)
    brach_data = torch.randn(batch_size, seq_len)
    
    # Create dummy masks (some missing values)
    aorta_mask = torch.rand(batch_size, seq_len) < 0.1  # 10% missing
    brach_mask = torch.rand(batch_size, seq_len) < 0.1  # 10% missing
    
    # Forward pass
    logits = model(aorta_data, brach_data, aorta_mask, brach_mask)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Expected: [{batch_size}, 6]")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
