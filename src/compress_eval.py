import argparse
import os
import torch
import torch.nn as nn
import wandb

from .utils import load_checkpoint, model_size_bytes
from .data import get_cifar10_loaders
from .models import MobileNetV2CIFAR10
from .quantization import QuantConfig, compress_mobilenetv2

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--project", type=str, default="cs6886_mobilenet_quant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # baseline model + size
    model = MobileNetV2CIFAR10()
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, test_loader = get_cifar10_loaders(args.data_root, batch_size=128)
    base_loss, base_acc = evaluate(model, test_loader, device)
    baseline_size_mb = model_size_bytes(model, 4) / (1024**2)

    print(f"Baseline -> loss={base_loss:.4f}, acc={base_acc*100:.2f}%, size={baseline_size_mb:.2f}MB")

    sweep_cfgs = [
        QuantConfig(weight_bits=8, act_bits=8, per_channel_weights=True),
        QuantConfig(weight_bits=6, act_bits=8, per_channel_weights=True),
        QuantConfig(weight_bits=6, act_bits=6, per_channel_weights=True),
        QuantConfig(weight_bits=4, act_bits=8, per_channel_weights=True),
        QuantConfig(weight_bits=4, act_bits=6, per_channel_weights=True),
        QuantConfig(weight_bits=4, act_bits=4, per_channel_weights=True),
        QuantConfig(weight_bits=8, act_bits=4, per_channel_weights=True),
        QuantConfig(weight_bits=8, act_bits=8, per_channel_weights=False),
    ]

    for idx, qcfg in enumerate(sweep_cfgs):
        run_name = f"quant_cfg_{idx}_w{qcfg.weight_bits}_a{qcfg.act_bits}_pc{int(qcfg.per_channel_weights)}"
        wandb.init(project=args.project, name=run_name, config={
            "weight_quant_bits": qcfg.weight_bits,
            "activation_quant_bits": qcfg.act_bits,
            "per_channel_weights": qcfg.per_channel_weights,
        })

        model_q = compress_mobilenetv2(model, qcfg)
        model_q.to(device)

        q_loss, q_acc = evaluate(model_q, test_loader, device)

        # 4-byte baseline vs (weight_bits/8)-byte compressed
        compressed_size_mb = baseline_size_mb * (qcfg.weight_bits / 32.0)
        compression_ratio = baseline_size_mb / compressed_size_mb

        wandb.log({
            "weight_bits": qcfg.weight_bits,
            "act_bits": qcfg.act_bits,
            "baseline_size_mb": baseline_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": compression_ratio,
            "quantized_acc": q_acc * 100.0,
        })

        print(
            f"[{run_name}] acc={q_acc*100:.2f}%  "
            f"size={compressed_size_mb:.2f}MB  ratio={compression_ratio:.2f}x"
        )

        wandb.finish()

if __name__ == "__main__":
    main()