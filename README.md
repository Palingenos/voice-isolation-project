# Voice Isolation

A web app that extracts clean vocals from any audio or video file and optionally generates lyrics with timestamps.

## Features

- Upload MP4, MP3, WAV, FLAC, OGG, or M4A files
- Isolate vocals from all background music and instruments
- Auto-generate lyrics with full-text and timestamped views
- Download the isolated vocals as a WAV file

## Requirements

- Python 3.9+
- ffmpeg

### Install ffmpeg (if not already installed)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

## Setup

1. Clone or download this project, then navigate to the project folder:

```bash
cd "Voice isolation project"
```

2. Install Python dependencies:

```bash
pip install fastapi uvicorn[standard] python-multipart aiofiles demucs openai-whisper
```

## Running the App

```bash
uvicorn app:app --reload
```

Then open **http://localhost:8000** in your browser.

> On first run, the app will automatically download the demucs model (~80 MB) and the Whisper base model (~74 MB). This only happens once.

## Usage

1. Drag and drop a file onto the upload area, or click to browse
2. Check **"Generate lyrics automatically"** if you want a transcript (on by default)
3. Click **Isolate Vocals**
4. Wait for processing to complete — status updates in real time
5. Play or download the isolated vocals
6. View lyrics in **Full text** or **Timed** (with timestamps) mode
7. Click **Copy** to copy the lyrics to your clipboard
8. Click **Start over** to process another file

## Processing Times (approximate, on CPU)

| File length | Vocal separation | Lyrics generation |
|-------------|-----------------|-------------------|
| 1 minute    | ~30 sec         | ~15 sec           |
| 3 minutes   | ~1–2 min        | ~30 sec           |
| 5 minutes   | ~3–4 min        | ~1 min            |

Processing is faster if you have a GPU (CUDA or Apple Silicon MPS).

## Project Structure

```
Voice isolation project/
├── app.py              # FastAPI backend
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
└── static/
    └── index.html      # Web UI
```

## How It Works

- **Vocal separation** — [Demucs](https://github.com/facebookresearch/demucs) by Facebook Research splits the audio into vocals and accompaniment using a neural network
- **Lyrics generation** — [OpenAI Whisper](https://github.com/openai/whisper) transcribes the isolated vocal track to text
- **Audio conversion** — ffmpeg handles MP4 extraction and format conversion before processing
