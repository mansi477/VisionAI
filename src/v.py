
import subprocess
import tempfile
import os
import winsound
from piper import PiperVoice

MODEL_PATH = "en_US-lessac-medium.onnx"

voice = PiperVoice.load(MODEL_PATH)

messages = [
    "Voice test started.",
    "Person detected, approximately two metres away, on your left.",
    "Move slightly to the right if the path is clear."
]

for message in messages:
    print("Speaking:", message)

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:
        wav_path = temp.name

    try:
        with open(wav_path, "wb") as wav_file:
            voice.synthesize_wav(message, wav_file)

        winsound.PlaySound(
            wav_path,
            winsound.SND_FILENAME
        )

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

print("All voice tests completed.")
