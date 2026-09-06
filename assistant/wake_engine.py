"""Use the in-tree wake implementation with bounded, sleeping CPU workers."""
from pathlib import Path
import sys


def create_wake_model(model_path):
    source = str(Path(__file__).resolve().parent.parent / "wakeword" / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import onnxruntime as ort
    from livekit.wakeword import WakeWordModel

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return WakeWordModel(models=[model_path], sess_options=options)
