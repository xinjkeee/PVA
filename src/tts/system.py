import shutil
import subprocess
from pathlib import Path


def synthesize_with_macos_say(
    text: str,
    output_path: Path,
    voice: str | None = None,
    rate: int | None = None,
) -> Path:
    say_path = shutil.which("say")
    if say_path is None:
        raise RuntimeError("macOS 'say' command is not available on this machine")
    if not text.strip():
        raise ValueError("text must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [say_path]
    if voice:
        command.extend(["-v", voice])
    if rate is not None:
        command.extend(["-r", str(rate)])
    command.extend(["-o", str(output_path), text])
    subprocess.run(command, check=True)
    return output_path


def play_audio_file(audio_path: Path) -> bool:
    afplay_path = shutil.which("afplay")
    if afplay_path is not None:
        subprocess.run([afplay_path, str(audio_path)], check=True)
        return True

    try:
        import sounddevice as sd
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Audio playback requires macOS 'afplay' or optional packages "
            "'sounddevice' and 'soundfile'."
        ) from exc

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    sd.play(audio, sample_rate)
    sd.wait()
    return True
