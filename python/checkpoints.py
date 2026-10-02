"""Load tensor-only legacy weights or metadata-bearing baseline checkpoints."""

import hashlib
import warnings
from pathlib import Path

import torch


# The released tensor-only checkpoint has no embedded field metadata. Identify
# its bytes, not its filename: users can move or rename the supplied weights.
# Source and verification scope are recorded in docs/checkpoint-provenance.json.
KNOWN_LEGACY_FIELDS = {
    "2b60b7dd3b94efe4dd8530d018cf93d2391854096c41e76ad7b1a7113f5cce5b": "concentration",
}


def load_checkpoint(path, model, device="cpu", field=None):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state dictionary or a baseline checkpoint dictionary.")
    if "physics_adapter_state_dict" in checkpoint:
        raise ValueError("This checkpoint contains an experimental adapter and is outside this baseline.")
    metadata = checkpoint if "model_state_dict" in checkpoint else {}
    recorded_field = metadata.get("config", {}).get("field")
    if not metadata and field:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        recorded_field = KNOWN_LEGACY_FIELDS.get(digest.hexdigest())
    if field and recorded_field and field != recorded_field:
        raise ValueError(f"Checkpoint was trained for {recorded_field}, not {field}.")
    if field and not recorded_field:
        warnings.warn(
            f"Checkpoint has no verified field metadata; cannot confirm it was trained for {field}. "
            "Only use weights whose prediction target you have independently confirmed.",
            UserWarning, stacklevel=2,
        )
    model.load_state_dict(metadata["model_state_dict"] if metadata else checkpoint, strict=True)
    return metadata
