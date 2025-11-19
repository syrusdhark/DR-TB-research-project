"""
Model loading utilities for DR-TB prediction.
Handles loading saved model checkpoints and initializing the model architecture.
"""

import torch
import os
from pathlib import Path
from model import MultimodalFusionModel
import config


def load_model(model_path=None, device=None):
    """
    Load a trained DR-TB prediction model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pth file). If None, uses latest.
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
    
    Returns:
        Loaded model in evaluation mode
    """
    # Auto-detect device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Get model path
    if model_path is None:
        model_path = config.get_latest_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"No model files found in {config.MODELS_DIR}. "
                "Please ensure model checkpoints exist."
            )
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    # Note: weights_only=False is safe for local trusted model files
    # PyTorch 2.6+ defaults to weights_only=True for security, but our models contain numpy objects
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}")
    
    # Extract model state and metadata
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            # Try to get feature dimensions from checkpoint
            num_clinical = checkpoint.get('num_clinical_features', config.NUM_CLINICAL_FEATURES)
            num_genomic = checkpoint.get('num_genomic_features', config.NUM_GENOMIC_FEATURES)
        else:
            # Assume entire dict is state dict
            model_state = checkpoint
            num_clinical = config.NUM_CLINICAL_FEATURES
            num_genomic = config.NUM_GENOMIC_FEATURES
    else:
        raise ValueError("Invalid checkpoint format")
    
    # Create model with correct architecture
    model = MultimodalFusionModel(
        num_clinical_features=num_clinical,
        num_genomic_features=num_genomic,
        num_classes=1
    )
    
    # Load state dict with flexible matching
    # Some models may have been saved with different architecture versions
    try:
        # Try strict loading first
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try flexible loading
        # This handles architecture differences between training and inference versions
        print("⚠️  Strict loading failed, attempting flexible loading...")
        print("   ℹ️  This usually means the model was saved with a different architecture version.")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            if missing_keys:
                print(f"   ⚠️  Missing keys (using random initialization): {len(missing_keys)} keys")
                # Only warn about critical missing keys
                critical_missing = [k for k in missing_keys if 'classifier' in k or 'fusion' in k]
                if critical_missing:
                    print(f"   ⚠️  WARNING: Critical fusion/classifier layers missing!")
                    print(f"      This may affect prediction accuracy.")
                    print(f"      Missing: {', '.join(critical_missing[:3])}...")
            if unexpected_keys:
                print(f"   ℹ️  Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys")
            print("   ✅ Model loaded with flexible matching (some layers use random weights)")
        except Exception as e2:
            raise RuntimeError(f"Failed to load model weights even with flexible loading: {e2}")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"   • Clinical features: {num_clinical}")
    print(f"   • Genomic features: {num_genomic}")
    
    return model, device


def get_model_info(model_path=None):
    """
    Get information about a model checkpoint without loading it.
    
    Args:
        model_path: Path to model checkpoint. If None, uses latest.
    
    Returns:
        Dictionary with model information
    """
    if model_path is None:
        model_path = config.get_latest_model_path()
    
    if model_path is None or not Path(model_path).exists():
        return None
    
    model_path = Path(model_path)
    info = {
        'path': str(model_path),
        'name': model_path.name,
        'size_mb': model_path.stat().st_size / (1024 * 1024),
        'modified': model_path.stat().st_mtime
    }
    
    # Try to load checkpoint metadata
    # Note: weights_only=False is safe for local trusted model files
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                info['num_clinical_features'] = checkpoint.get('num_clinical_features', config.NUM_CLINICAL_FEATURES)
                info['num_genomic_features'] = checkpoint.get('num_genomic_features', config.NUM_GENOMIC_FEATURES)
                if 'validation_auc' in checkpoint:
                    info['validation_auc'] = checkpoint['validation_auc']
                if 'validation_f1' in checkpoint:
                    info['validation_f1'] = checkpoint['validation_f1']
    except:
        pass
    
    return info

