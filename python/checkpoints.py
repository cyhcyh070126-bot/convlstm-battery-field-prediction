"""Load tensor-only legacy weights or metadata-bearing baseline checkpoints."""

import torch


def load_checkpoint(path, model, device="cpu", field=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state dictionary or a baseline checkpoint dictionary.")
    if "physics_adapter_state_dict" in checkpoint:
        raise ValueError("This checkpoint contains an experimental adapter and is outside this baseline.")
    metadata = checkpoint if "model_state_dict" in checkpoint else {}
    recorded_field = metadata.get("config", {}).get("field")
    if field and recorded_field and field != recorded_field:
        raise ValueError(f"Checkpoint was trained for {recorded_field}, not {field}.")
    model.load_state_dict(metadata["model_state_dict"] if metadata else checkpoint, strict=True)
    return metadata
