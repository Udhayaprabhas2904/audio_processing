# Human Voice Extraction API

A FastAPI-based backend for extracting and enhancing human speech from noisy audio recordings.

The system accepts an audio file, detects portions containing human speech, removes non-speech sections, and applies speech enhancement to suppress background noise while preserving the original recorded voice as much as possible.

---

## Overview

Real-world audio recordings can contain unwanted background noise, silence, music, environmental sounds, and other non-speech sections.

This project provides an automated audio-processing pipeline focused on **human speech extraction and enhancement**.

The API:

1. Accepts an audio file.
2. Converts the input into mono PCM WAV using FFmpeg.
3. Detects human speech using Silero VAD.
4. Removes non-speech portions.
5. Enhances the detected speech using SpeechBrain.
6. Restores the output to the original sample rate.
7. Returns the processed speech as a WAV file.

The project is designed as a backend API and can be integrated with web applications, desktop applications, or other services.

---

## Key Features

* Human speech detection from noisy recordings
* Non-speech segment removal
* Background noise suppression
* Multiple audio format support
* Automatic audio format conversion using FFmpeg
* FastAPI REST API
* Swagger/OpenAPI documentation
* CPU-based processing
* Original recorded speech is preserved rather than generating synthetic speech
* Automatic temporary file cleanup
* Processing latency measurement

---

## Technologies Used

* **Python**
* **FastAPI**
* **Uvicorn**
* **Silero VAD**
* **SpeechBrain**
* **PyTorch**
* **TorchAudio**
* **FFmpeg**
* **SoundFile**
* **NumPy**

---

## Processing Pipeline

```text
                    Input Audio
                         |
                         v
                +----------------+
                |     FFmpeg     |
                | Audio Conversion|
                +----------------+
                         |
                         v
                  Mono PCM WAV
                         |
                         v
                +----------------+
                |   Silero VAD   |
                | Speech Detection|
                +----------------+
                         |
                         v
              Human Speech Regions
                         |
                         v
              Remove Non-Speech
                         |
                         v
                +----------------+
                |   SpeechBrain  |
                |    Enhancement |
                +----------------+
                         |
                         v
                 Enhanced Speech
                         |
                         v
              Restore Sample Rate
                         |
                         v
                   WAV Output
```

---

## Supported Audio Formats

The API supports the following input formats:

```text
.wav
.mp3
.m4a
.aac
.flac
.ogg
.oga
.opus
.wma
.aiff
.aif
.au
.webm
.mp4
.mka
.caf
```

The uploaded file is converted to **mono PCM WAV** before audio processing.

---

## Project Structure

```text
voice_extraction/
│
├── backend/
│   ├── main.py
│   └── audio_processor.py
│
├── uploads/
├── converted/
├── outputs/
├── pretrained_models/
│   └── speechbrain_enhancement/
│
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

---

## Run the API

From the project root directory:

```powershell
uvicorn backend.main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

---

## Swagger Documentation

Open the following URL in a browser:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to upload an audio file and test the API.

---

## API Endpoint

### Process Audio

```text
POST /process-audio
```

### Input

Upload an audio file using the `file` parameter.

### Output

The API returns:

```text
human_voice.wav
```

with:

```text
Content-Type: audio/wav
```

### Successful Response

```text
200 OK
```

---

## Response Codes

| Code | Description                       |
| ---- | --------------------------------- |
| 200  | Audio processed successfully      |
| 400  | Invalid or unsupported audio file |
| 422  | Request validation error          |
| 500  | Audio processing failed           |

---

## Audio Processing

### 1. FFmpeg Conversion

The uploaded audio is converted into a standard **mono PCM WAV** format so that it can be processed consistently.

### 2. Silero VAD

Silero Voice Activity Detection identifies portions of the audio containing human speech.

Non-speech sections such as silence and other detected non-speech regions are removed.

### 3. Speech Extraction

The detected speech regions are combined to create an audio file containing only the identified human speech.

### 4. Speech Enhancement

SpeechBrain enhancement is applied to reduce background noise and improve speech quality while preserving the original recorded voice.

### 5. Output

The processed speech is saved as a WAV file and returned through the API.

---

## Performance Testing

The application measures processing latency using `time.perf_counter()`.

Example test:

```text
Original duration: 949.08s
Speech duration: 23.90s
Removed approximately: 925.17s
Output sample rate: 44100
Output size: 2108378 bytes
Processing latency: 63.00 seconds
```

### Performance Summary

| Metric             |         Result |
| ------------------ | -------------: |
| Input duration     | 949.08 seconds |
| Detected speech    |  23.90 seconds |
| Non-speech removed | 925.17 seconds |
| Output sample rate |      44,100 Hz |
| Output size        |         2.1 MB |
| Processing latency |     63 seconds |

> Processing latency depends on the input audio duration, amount of detected speech, model processing, and available CPU resources.

---

## Temporary File Cleanup

Uploaded files and intermediate converted WAV files are automatically removed after processing.

This prevents unnecessary storage of temporary files.

---

## API Flow

```text
Client
  |
  | Upload Audio
  v
FastAPI
  |
  v
Validate File
  |
  v
FFmpeg Conversion
  |
  v
Silero VAD
  |
  v
Extract Speech
  |
  v
SpeechBrain Enhancement
  |
  v
Save WAV
  |
  v
Return human_voice.wav
```

---

## Limitations

* Processing is currently CPU-based.
* Processing time depends on the input audio length and system hardware.
* Background voices may not always be completely removed if they overlap with the target speaker.
* Speech enhancement improves audio quality but cannot guarantee perfect separation of overlapping speakers.
* The system extracts and enhances existing speech; it does not generate synthetic speech.

---

## Future Improvements

* GPU acceleration for faster processing
* Improved speaker separation
* Better overlapping-speaker handling
* Real-time audio processing
* Audio quality metrics
* Support for speaker diarization
* Web-based frontend
* Batch audio processing
* Processing progress/status tracking

---

## Status

**Project Status: Completed**

The Human Voice Extraction API has been implemented and tested successfully using FastAPI, FFmpeg, Silero VAD, and SpeechBrain.
