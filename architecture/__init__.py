"""
Model registry for RGB → Hyperspectral reconstruction.

To add a new model:
  1. Put the file in architecture/
  2. Import the class below and add it to _MODELS

build_model() automatically filters kwargs to only pass what each
model's __init__ accepts, so shared args (n_feat, stage, ...) work
across models without manual per-model dispatch.
"""

import inspect
from .MSTpp import MSTpp
# from .LWMSR import LWMSR

_MODELS = {
    'mstpp': MSTpp,
    # 'lwmsr': LWMSR,
}


def build_model(name: str, in_channels: int, out_channels: int, **kwargs):
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f'Unknown model "{name}". Available: {list(_MODELS)}')
    cls = _MODELS[name]
    valid = set(inspect.signature(cls.__init__).parameters) - {'self'}
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(in_channels=in_channels, out_channels=out_channels, **filtered)


def list_models():
    return list(_MODELS)
