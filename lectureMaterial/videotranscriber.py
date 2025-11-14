"""
Transcribe all .mp4 files in a directory using Faster Whisper.

Requires: pip install faster-whisper and ffmpeg in PATH.

Usage: python main.py [folder] [model] [device]
Example: python main.py video_directory small cuda
"""

import sys
import shutil
from pathlib import Path
from faster_whisper import WhisperModel


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


def get_device(device: str = None) -> str:
    """Auto-detect device or use specified one."""
    if device:
        return device
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    
    return "cpu"


def transcribe_all(folder: str, model: str = "small", device: str = None) -> None:
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Folder not found: {folder}")
        return
    
    if not check_ffmpeg():
        print("Error: ffmpeg not found in PATH. Please install ffmpeg and add it to PATH.")
        return
    
    mp4_files = list(folder_path.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in {folder}")
        return
    
    detected_device = get_device(device)
    output_dir = Path("transcriptions")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Found {len(mp4_files)} MP4 files. Starting transcription...")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Using device: {detected_device}")
    print(f"Loading model: {model}...\n")
    
    model_instance = WhisperModel(
        model,
        device=detected_device,
        compute_type="float16" if detected_device == "cuda" else "int8",
    )
    
    for mp4 in mp4_files:
        output_file = output_dir / f"{mp4.stem}.txt"
        
        if output_file.exists():
            print(f"Skipping {mp4.name} (already transcribed)")
            continue
        
        print(f"Transcribing: {mp4.name}...", end=" ", flush=True)
        
        try:
            segments, info = model_instance.transcribe(
                str(mp4),
                language="en",
                beam_size=1,
                best_of=1,
            )
            
            text = "\n".join([segment.text for segment in segments])
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            size = output_file.stat().st_size
            print(f"Done ({size:,} bytes)")
        except Exception as e:
            error_log = output_dir / f"{mp4.stem}_error.txt"
            with open(error_log, "w", encoding="utf-8") as f:
                f.write(f"Error transcribing: {mp4.name}\n\n")
                f.write(f"Error: {str(e)}\n")
            print(f"Error - details saved to {error_log.name}")
    
    print("\nTranscription complete.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "video_directory"
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    device = sys.argv[3] if len(sys.argv) > 3 else None
    transcribe_all(folder, model, device)
