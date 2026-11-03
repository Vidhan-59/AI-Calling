"""Convert Hugging Face Whisper fine-tunes (e.g. the IIT Madras "Whisper-Indic"
Hindi/Gujarati models) to CTranslate2, the format faster-whisper runs.

Needs torch + transformers, which the live server does NOT need - so run it
from a separate environment and keep the server's dependencies lean:

    python -m venv %USERPROFILE%\.cache\ai-calling\cvenv   (keep the path short:
    ...\cvenv\Scripts\pip install torch transformers ctranslate2   torch's DLLs fail
    ...\cvenv\Scripts\python tools\convert_stt_models.py          on long paths)

Output goes to ~/.cache/ai-calling/stt_models/<name>/ (STT_MODELS_DIR
overrides), which tools/stt_benchmark.py and LOCAL_STT_MODEL can point at.
"""

import os
import sys

from ctranslate2.converters import TransformersConverter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Multi-GB output - kept off the repo drive by default (STT_MODELS_DIR overrides).
OUT = os.environ.get("STT_MODELS_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "ai-calling", "stt_models")

MODELS = {
    "whisper-hindi-medium": "vasista22/whisper-hindi-medium",
    "whisper-gujarati-medium": "vasista22/whisper-gujarati-medium",
    "whisper-hindi-large-v2": "vasista22/whisper-hindi-large-v2",
}


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        dest = os.path.join(OUT, name)
        if os.path.isfile(os.path.join(dest, "model.bin")):
            print(f"{name}: already converted")
            continue
        print(f"{name}: converting {MODELS[name]} ...", flush=True)
        # float16 on disk; faster-whisper re-quantizes at load (int8_float16 etc.).
        TransformersConverter(MODELS[name], copy_files=["preprocessor_config.json"]).convert(
            dest, quantization="float16", force=True
        )
        # These repos ship only the slow tokenizer files; faster-whisper needs
        # tokenizer.json (without it, it silently downloads a stock one).
        from transformers import WhisperTokenizerFast

        WhisperTokenizerFast.from_pretrained(MODELS[name]).backend_tokenizer.save(os.path.join(dest, "tokenizer.json"))
        print(f"{name}: done -> {dest}", flush=True)


if __name__ == "__main__":
    main()
