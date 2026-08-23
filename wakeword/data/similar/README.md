# data/similar - hard negative recordings

This folder holds sounds that resemble **cozy** but must NOT trigger it:
nosy, rosy, Josie, Ozzie, dozy, posy, noisy, "cause he" ...

    synth_<word>_000.wav   seed clips committed here for illustration

The complete synthetic sets (hundreds per word) are generated into
work/similar/<word>/ at training time. Regenerate seeds or full sets:

    cd wakeword
    source .venv/bin/activate
    python generate_data.py --only similar            # fills missing sets
    python generate_data.py --only similar --force    # regenerate

Edit the similar_words list in ../config.yaml to teach Cozy new lookalikes.
