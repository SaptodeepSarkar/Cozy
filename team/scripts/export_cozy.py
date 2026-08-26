
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

WW = Path(__file__).resolve().parents[2] / "wakeword"
CFG = yaml.safe_load((WW / "config.yaml").read_text())
W = int(CFG["training"]["window_frames"])
EMB = 96
NAME = CFG["model"]["name"]


class WWNet(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        logits = self.net.net(flat)  # [B, 1] rank required by openWakeWord
        return torch.sigmoid(logits)


net = WWNet(W * EMB)
net.load_state_dict(torch.load(WW / "models" / (NAME + ".pt"),
                              map_location="cpu"))
wrapper = ExportWrapper(net).eval().float()
dummy = torch.zeros(1, W, EMB)

for p in WW.joinpath("models").glob(NAME + ".onnx*"):
    p.unlink()

onnx_path = WW / "models" / (NAME + ".onnx")
torch.onnx.export(
    wrapper, dummy, str(onnx_path),
    input_names=["input"], output_names=["scores"],
    opset_version=13,
    dynamo=False,
)
print("exported:", onnx_path.stat().st_size, "bytes")

try:
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(m, str(onnx_path))
except Exception as exc:
    print("consolidation skipped:", exc)

print("EXPORT_DONE")
