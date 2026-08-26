
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
WW = ROOT / "wakeword"
LOG = WW / "work" / "full.log"
SCRIPTS = pathlib.Path(__file__).resolve().parent
VENV_PY = WW / ".venv" / "bin" / "python"
REPORT = ROOT / "team" / "wake_final_report.txt"


def run(cmd, timeout):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, cwd=str(WW))


def segment():
    text = LOG.read_text(errors="ignore")
    idx = text.rfind("=== FULL RUN")
    return text[idx:] if idx >= 0 else ""


def main():
    deadline = time.time() + 520
    while time.time() < deadline:
        seg = segment()
        if "=== FULL_OK" in seg:
            lines = ["STATUS DONE_OK"]

            e = run(str(VENV_PY) + " " + str(SCRIPTS / "export_cozy.py"), 240)
            lines.append("EXPORT rc=" + str(e.returncode))
            lines += (e.stdout or "").splitlines()[-3:]

            threshold = 0.40
            mf = WW / "models" / "metrics.json"
            if mf.exists():
                try:
                    data = json.loads(mf.read_text())
                    threshold = float(data.get("safe_threshold_zero_fpr",
                                               0.40))
                except Exception:
                    pass

            a = run(str(VENV_PY) + " " + str(SCRIPTS / "score_cozy.py")
                    + " " + str(threshold), 400)
            lines.append("ACCEPTANCE rc=" + str(a.returncode)
                         + " threshold=" + str(threshold))
            lines += (a.stdout or "").splitlines()

            g = run("git add -A && git commit -q -m 'model: v13/v15 "
                    "hey-cozy wake word trained on cleaned normalized "
                    "recordings' && git push origin main", 180)
            lines.append("GIT rc=" + str(g.returncode))

            REPORT.write_text("\n".join(lines))
            print("FINALIZED - see team/wake_final_report.txt")
            return
        if "FULL_FAILED" in seg:
            tail = seg.splitlines()[-25:]
            REPORT.write_text("STATUS DONE_FAIL\n" + "\n".join(tail))
            print("FAILED - details in team/wake_final_report.txt")
            return
        time.sleep(20)
    print("STATUS RUNNING")


main()
