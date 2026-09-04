
import sys
from pathlib import Path

WW = Path(__file__).resolve().parents[2] / "wakeword"
text = open(WW / "work" / "full.log",
            errors="ignore").read()
marker = sys.argv[1] if len(sys.argv) > 1 else "COZYNET TRAIN v4 START"
i = text.rfind(marker)
if i < 0:
    print("MARKER NOT FOUND:", marker)
else:
    print(text[i:i + 3500])
