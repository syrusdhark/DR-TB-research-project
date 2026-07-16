"""
Model loading utilities for the two-stage TB / DR-TB pipeline.

Stage 1: TBImageClassifier   (CXR -> TB vs Normal)
Stage 2: DRTBRiskModel       (clinical + genomic -> DR-TB risk)
"""

import torch
from pathlib import Path
from model import TBImageClassifier, DRTBRiskModel
import config


def _resolve_device(device=None):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _load_checkpoint(model_path, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        # weights_only=False is safe for local trusted model files
        return torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {e}")


def load_tb_classifier(model_path=None, device=None):
    """Load the Stage 1 TB image classifier (CXR -> TB vs Normal)."""
    device = _resolve_device(device)

    if model_path is None:
        model_path = config.get_latest_model_path(config.TB_MODEL_PREFIX)
        if model_path is None:
            raise FileNotFoundError(
                f"No TB classifier checkpoint found in {config.MODELS_DIR} "
                f"(expected a file starting with '{config.TB_MODEL_PREFIX}')."
            )

    print(f"Loading TB image classifier from: {model_path}")
    checkpoint = _load_checkpoint(model_path, device)
    model_state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    model = TBImageClassifier(num_classes=1)
    model.load_state_dict(model_state, strict=True)
    model = model.to(device)
    model.eval()

    threshold = config.get_threshold_for_model(model_path, config.DEFAULT_TB_THRESHOLD)
    print(f"[OK] TB classifier loaded on {device} (threshold={threshold})")
    return model, device, threshold


def load_drtb_risk_model(model_path=None, device=None):
    """Load the Stage 2 DR-TB risk model (clinical + genomic -> risk)."""
    device = _resolve_device(device)

    if model_path is None:
        model_path = config.get_latest_model_path(config.DRTB_RISK_MODEL_PREFIX)
        if model_path is None:
            raise FileNotFoundError(
                f"No DR-TB risk model checkpoint found in {config.MODELS_DIR} "
                f"(expected a file starting with '{config.DRTB_RISK_MODEL_PREFIX}')."
            )

    print(f"Loading DR-TB risk model from: {model_path}")
    checkpoint = _load_checkpoint(model_path, device)
    model_state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    model = DRTBRiskModel(
        num_clinical_features=config.NUM_CLINICAL_FEATURES,
        num_genomic_features=config.NUM_GENOMIC_FEATURES,
        num_classes=1
    )
    model.load_state_dict(model_state, strict=True)
    model = model.to(device)
    model.eval()

    threshold = config.get_threshold_for_model(model_path, config.DEFAULT_DRTB_RISK_THRESHOLD)
    print(f"[OK] DR-TB risk model loaded on {device} (threshold={threshold})")
    return model, device, threshold


def get_model_info(prefix):
    """Get information about the latest checkpoint matching a prefix, without loading it."""
    model_path = config.get_latest_model_path(prefix)
    if model_path is None or not Path(model_path).exists():
        return None

    model_path = Path(model_path)
    info = {
        'path': str(model_path),
        'name': model_path.name,
        'size_mb': model_path.stat().st_size / (1024 * 1024),
        'modified': model_path.stat().st_mtime,
    }

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ('validation_auc', 'validation_f1', 'validation_accuracy'):
                if key in checkpoint:
                    info[key] = checkpoint[key]
    except Exception:
        pass

    return info
