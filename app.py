import os
import ssl
import uuid
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import certifi

import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice Isolation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# In-memory job store
jobs: dict[str, dict] = {}

whisper_model: Optional[whisper.Whisper] = None
whisper_lock = threading.Lock()


def get_whisper_model():
    global whisper_model
    with whisper_lock:
        if whisper_model is None:
            whisper_model = whisper.load_model("base")
    return whisper_model


def extract_audio_from_mp4(input_path: Path, output_path: Path):
    """Extract audio track from MP4 using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "44100", "-ac", "2", str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def run_demucs(audio_path: Path, job_dir: Path) -> Path:
    """Run demucs vocal separation. Returns path to vocals file."""
    env = os.environ.copy()
    env["SSL_CERT_FILE"] = certifi.where()
    env["REQUESTS_CA_BUNDLE"] = certifi.where()
    result = subprocess.run(
        ["python3", "-m", "demucs", "--two-stems=vocals",
         "-o", str(job_dir), str(audio_path)],
        capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed: {result.stderr}")

    # demucs outputs to: job_dir/htdemucs/stem_name/vocals.wav
    # Find the vocals file
    for vocals_file in job_dir.rglob("vocals.wav"):
        return vocals_file

    raise RuntimeError("demucs did not produce a vocals.wav file")


def process_job(job_id: str, upload_path: Path, filename: str, generate_lyrics: bool):
    """Background task: separate vocals and optionally transcribe."""
    job = jobs[job_id]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    try:
        # Step 1: Convert to WAV if needed
        job["status"] = "extracting_audio"
        suffix = upload_path.suffix.lower()
        if suffix == ".mp4":
            audio_path = job_dir / "audio.wav"
            extract_audio_from_mp4(upload_path, audio_path)
        elif suffix in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            audio_path = job_dir / "audio.wav"
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(upload_path),
                 "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                 str(audio_path)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Step 2: Separate vocals
        job["status"] = "separating_vocals"
        vocals_path = run_demucs(audio_path, job_dir)

        # Copy vocals to a known location
        final_vocals = job_dir / "vocals_final.wav"
        shutil.copy(vocals_path, final_vocals)
        job["vocals_path"] = str(final_vocals)

        # Step 3: Generate lyrics (optional)
        if generate_lyrics:
            job["status"] = "generating_lyrics"
            model = get_whisper_model()
            transcript = model.transcribe(str(final_vocals))
            job["lyrics"] = transcript["text"].strip()
            job["segments"] = [
                {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in transcript["segments"]
            ]

        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        # Clean up upload
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    generate_lyrics: bool = True,
):
    allowed = {".mp4", ".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {', '.join(allowed)}")

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"

    # Save upload, enforcing size limit
    size = 0
    try:
        with open(upload_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            while chunk := file.file.read(chunk_size):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    f.close()
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(413, f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(500, "Failed to save uploaded file.")

    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "status": "queued",
        "lyrics": None,
        "segments": None,
        "error": None,
        "vocals_path": None,
    }

    background_tasks.add_task(
        process_job, job_id, upload_path, file.filename, generate_lyrics
    )

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "lyrics": job["lyrics"],
        "segments": job["segments"],
        "error": job["error"],
        "has_vocals": job["vocals_path"] is not None,
    }


@app.get("/api/download/{job_id}/vocals")
def download_vocals(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "done" or not job["vocals_path"]:
        raise HTTPException(400, "Vocals not ready yet")
    vocals_path = Path(job["vocals_path"])
    if not vocals_path.exists():
        raise HTTPException(404, "Vocals file missing")

    stem = Path(job["filename"]).stem
    return FileResponse(
        vocals_path,
        media_type="audio/wav",
        filename=f"{stem}_vocals.wav",
    )


app.mount("/", StaticFiles(directory="static", html=True), name="static")
