# Personal Voice Assistant

End-to-end Russian-language voice assistant pipeline built from scratch.

## Pipeline

**VAD → KWS → ASR → NLU → TTS**

| Module | Description |
|--------|-------------|
| VAD | Energy-based voice activity detection (custom) + WebRTC VAD |
| KWS | Lightweight CNN on mel spectrograms, wake word "привет" |
| ASR | OpenAI Whisper with WER/CER evaluation |
| NLU | Rule-based intent recognition |
| TTS | macOS `say` wrapper |

## Usage

```bash
pip install -r requirements.txt

# Process audio
python main.py process --input-dir ./data/raw

# Train keyword spotter
python main.py train-kws --keyword "привет" --train-manifest ./data/kws/kws_train.jsonl

# Run assistant
python main.py assistant-gui
python main.py assistant-file --audio input.wav
```

## Dataset

Mozilla Common Voice (Russian).
