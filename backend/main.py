from pathlib import Path
import shutil
import uuid
import subprocess
import time

import imageio_ffmpeg

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.responses import FileResponse, Response

from backend.audio_processor import process_audio


# FASTAPI

app = FastAPI(
    title="Human Voice Extraction API",
    description=(
        "Upload an audio file and extract only "
        "the portions containing human speech "
        "while suppressing background noise."
    ),
    version="1.0.0",
)


# DIRECTORIES

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
CONVERTED_DIR = BASE_DIR / "converted"
OUTPUT_DIR = BASE_DIR / "outputs"


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONVERTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# BUNDLED FFMPEG

try:

    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

    print()

    print("FFmpeg executable:")
    print(FFMPEG_EXE)

except Exception as e:

    FFMPEG_EXE = None

    print()

    print("WARNING: Could not locate bundled FFmpeg.")
    print("Error:", e)


# ALLOWED AUDIO EXTENSIONS

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".au",
    ".webm",
    ".mp4",
    ".mka",
    ".caf",
}


# CHECK BUNDLED FFMPEG

def check_ffmpeg():
   

    if not FFMPEG_EXE:
        return False

    try:

        result = subprocess.run(
            [
                FFMPEG_EXE,
                "-version",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )

        return result.returncode == 0

    except (
        FileNotFoundError,
        subprocess.SubprocessError,
        OSError,
    ):

        return False


# CONVERT ANY AUDIO TO WAV

def convert_to_wav(
    input_file,
    output_file,
):
    

    print()

    print("CONVERTING AUDIO TO WAV")

    print(
        "Input:",
        input_file,
    )

    print(
        "Output:",
        output_file,
    )

    # Check FFmpeg

    if not check_ffmpeg():

        raise RuntimeError(
            "Bundled FFmpeg is not available. "
            "Make sure imageio-ffmpeg is installed with:\n"
            "python -m pip install imageio-ffmpeg"
        )

    # FFmpeg command

    command = [
        FFMPEG_EXE,

        "-y",

        "-i",
        str(input_file),

        # Convert to mono
        "-ac",
        "1",

        # Standard PCM WAV
        "-c:a",
        "pcm_s16le",

        str(output_file),
    ]

    print(
        "Running bundled FFmpeg..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check FFmpeg result

    if result.returncode != 0:

        print()

        print("FFMPEG ERROR")

        print(result.stderr)

        raise RuntimeError(
            "FFmpeg could not convert the uploaded audio."
        )

    # Verify output

    output_path = Path(output_file)

    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg completed but the WAV "
            "file was not created."
        )

    if output_path.stat().st_size == 0:

        raise RuntimeError(
            "FFmpeg created an empty WAV file."
        )

    print(
        "Audio conversion completed."
    )


# HOME

@app.get("/")
def home():

    return {
        "message": (
            "Human Voice Extraction API is running"
        ),

        "ffmpeg": (
            "bundled"
            if FFMPEG_EXE
            else "not available"
        ),
    }


# FFMPEG STATUS

@app.get("/ffmpeg")
def ffmpeg_status():

    working = check_ffmpeg()

    return {
        "available": working,
        "executable": FFMPEG_EXE,
        "source": "imageio-ffmpeg",
    }


# PROCESS AUDIO

class AudioWAVResponse(Response):
    media_type = "audio/wav"


@app.post(
    "/process-audio",
    response_class=AudioWAVResponse,
    responses={
        200: {
            "description": "Processed human speech WAV file.",
            "content": {
                "audio/wav": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    }
                }
            },
        },
        400: {
            "description": "Invalid or unsupported audio file.",
        },
        500: {
            "description": "Audio processing failed.",
        },
    },
)
async def process_uploaded_audio(
    file: UploadFile = File(...)
):

    # START TOTAL API LATENCY TIMER

    start_time = time.perf_counter()

    print()
    
    print("STARTING AUDIO PROCESSING")
  

    # VALIDATE FILENAME

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No audio file selected.",
        )

    # GET FILE EXTENSION

    original_filename = Path(
        file.filename
    )

    file_extension = (
        original_filename.suffix.lower()
    )

    if not file_extension:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file has no "
                "audio file extension."
            ),
        )

    # CHECK SUPPORTED EXTENSION

    if (
        file_extension
        not in ALLOWED_AUDIO_EXTENSIONS
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: "
                f"{file_extension}. "

                "Supported formats include "
                "WAV, MP3, M4A, AAC, FLAC, "
                "OGG, OPUS, WMA, AIFF, "
                "WEBM, MP4 and more."
            ),
        )

    # CREATE UNIQUE FILE ID

    file_id = str(
        uuid.uuid4()
    )

    # ORIGINAL UPLOADED FILE

    input_path = (
        UPLOAD_DIR
        / f"{file_id}{file_extension}"
    )

    # TEMPORARY INTERNAL WAV

    converted_path = (
        CONVERTED_DIR
        / f"{file_id}.wav"
    )

    # FINAL OUTPUT

    output_path = (
        OUTPUT_DIR
        / f"{file_id}_voice.wav"
    )

    try:

        # 1. SAVE UPLOADED FILE

        with open(
            input_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        print()

        print("UPLOADED FILE")

        print(
            "Original filename:",
            file.filename,
        )

        print(
            "Input path:",
            input_path,
        )

        # 2. CONVERT TO WAV

        convert_to_wav(
            input_path,
            converted_path,
        )

        # 3. PROCESS HUMAN SPEECH

        print()

        print(
            "STARTING HUMAN SPEECH EXTRACTION"
        )

        process_audio(
            str(converted_path),
            str(output_path),
        )

        # 4. CHECK OUTPUT

        if not output_path.exists():

            raise RuntimeError(
                "Audio processing failed. "
                "Output file was not created."
            )

        if output_path.stat().st_size == 0:

            raise RuntimeError(
                "Audio processing created "
                "an empty output file."
            )

        print()

        print("OUTPUT CREATED")

        print(
            "Output path:",
            output_path,
        )

        print(
            "Output size:",
            output_path.stat().st_size,
            "bytes",
        )

        # CALCULATE TOTAL API LATENCY

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        print()

       

        print(
            f"TOTAL API PROCESSING LATENCY: "
            f"{latency:.2f} seconds"
        )

        

        print()

        print("PROCESSING COMPLETE")

        # 5. RETURN CLEANED AUDIO

        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename="human_voice.wav",
        )

    # PRESERVE FASTAPI ERRORS

    except HTTPException:

        raise

    # HANDLE PROCESSING ERRORS

    except Exception as e:

        print()

        print("AUDIO PROCESSING ERROR")

        print(
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Audio processing failed: {str(e)}"
            ),
        )

    # CLEANUP TEMPORARY FILES

    finally:

        # Close uploaded file

        try:

            await file.close()

        except Exception:

            pass

        # Delete temporary converted WAV

        if converted_path.exists():

            try:

                converted_path.unlink()

                print(
                    "Temporary converted WAV deleted."
                )

            except Exception as e:

                print(
                    "Could not delete temporary WAV:",
                    e,
                )

        # Delete original uploaded file

        if input_path.exists():

            try:

                input_path.unlink()

                print(
                    "Temporary uploaded file deleted."
                )

            except Exception as e:

                print(
                    "Could not delete uploaded file:",
                    e,
                )
