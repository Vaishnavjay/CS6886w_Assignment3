from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2

@dataclass
class QuantConfig:
    weight_bits: int = 8
    act_bits: int = 8
    per_channel_weights: bool = True
    symmetric: bool = True
    act_clip: float = 6.0


class QuantAct(nn.Module):
    def __init__(self, num_bits: int = 8, act_clip: float = 6.0, symmetric: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.act_clip = act_clip
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_bits is None:
            return x

        x = torch.clamp(x, -self.act_clip, self.act_clip)
        qmax = 2 ** (self.num_bits - 1) - 1  # symmetric signed

        max_val = x.abs().max().detach() + 1e-8
        scale = max_val / qmax

        x_int = torch.round(x / scale)
        x_int = torch.clamp(x_int, -qmax, qmax)
        x_deq = x_int * scale
        return x_deq


def _quantize_tensor_per_tensor(w: torch.Tensor, num_bits: int):
    qmax = 2 ** (num_bits - 1) - 1
    max_val = w.abs().max().detach() + 1e-8
    scale = max_val / qmax
    w_int = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    zp = torch.zeros(1, dtype=torch.int8)
    return w_int, scale.view(1), zp


def _quantize_tensor_per_channel(w: torch.Tensor, num_bits: int):
    qmax = 2 ** (num_bits - 1) - 1
    max_val = w.abs().amax(dim=(1, 2, 3), keepdim=True).detach() + 1e-8
    scale = max_val / qmax
    w_int = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    zp = torch.zeros(w.size(0), dtype=torch.int8)
    return w_int, scale.squeeze(), zp


def quantize_module_weights(m: nn.Module, qcfg: QuantConfig):
    if not hasattr(m, "weight") or m.weight is None:
        return

    with torch.no_grad():
        w = m.weight.data

        if isinstance(m, nn.Conv2d) and qcfg.per_channel_weights:
            w_int, scale, zp = _quantize_tensor_per_channel(w, qcfg.weight_bits)
        else:
            w_int, scale, zp = _quantize_tensor_per_tensor(w, qcfg.weight_bits)

        m.register_buffer("weight_int", w_int)
        m.register_buffer("weight_scale", scale)
        m.register_buffer("weight_zp", zp)


def compress_mobilenetv2(model: nn.Module, qcfg: QuantConfig) -> nn.Module:
    model_q = copy.deepcopy(model).cpu().eval()

    # weight quant
    for _, m in model_q.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            quantize_module_weights(m, qcfg)

    # activation quant: before classifier
    if hasattr(model_q, "mnet") and isinstance(model_q.mnet, MobileNetV2):
        old_cls = model_q.mnet.classifier
        model_q.mnet.classifier = nn.Sequential(
            QuantAct(
                num_bits=qcfg.act_bits,
                act_clip=qcfg.act_clip,
                symmetric=qcfg.symmetric,
            ),
            *list(old_cls),
        )

    print("Finished building compressed MobileNetV2 (weights + activations).")
    return model_q