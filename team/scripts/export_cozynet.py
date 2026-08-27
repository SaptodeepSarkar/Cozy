
import sys
from pathlib import Path

import torch
import torch.nn as nn

WW = Path(__file__).resolve().parents[2] / "wakeword"
meta = __import__("json").loads((WW / "models" / "cozynet_v1_meta.json").read_text())
T = meta["time_frames"]


class CozyNet(nn.Module):
    def __init__(self):
        super().__init__()
        l1 = T // 4
        l2 = l1 + 1
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d((4, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.ZeroPad2d((0, 0, 1, 0)),
            nn.MaxPool2d((l2, 1)),
            nn.Flatten(),
            nn.Linear(32 * 32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = CozyNet()
model.load_state_dict(torch.load(WW / "models" / "cozynet_v1.pt",
                                 map_location="cpu"))
model.eval()

onnx_path = WW / "models" / "cozynet_v1.onnx"
torch.onnx.export(model, torch.zeros(1, 1, T, 32), str(onnx_path),
                  input_names=["input"], output_names=["scores"],
                  opset_version=13, dynamo=False)
print("exported:", onnx_path.stat().st_size)
print("EXPORT_DONE")
