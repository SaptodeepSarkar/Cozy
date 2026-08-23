# data/cozy - wake word recordings

This folder holds recordings of the word **cozy** used as POSITIVE training
data.

    recording_000.wav   your own microphone clips (record_samples.py)
    synth_000.wav       tiny synthetic demo set copied by generate_data.py

Add more of your voice any time:

    cd wakeword
    source .venv/bin/activate
    python record_samples.py --num 20

Tips: vary tone, speed, distance from the mic and background noise.
Clips are stored as 16 kHz mono WAV automatically. Retrain afterwards:

    python generate_data.py --mode full
    python train_wakeword.py
