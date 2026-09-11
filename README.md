# Human Voice Extraction API

A FastAPI-based backend for extracting human speech from noisy audio recordings.

The system accepts an audio file, detects the portions containing human speech, removes non-speech sections, and applies speech enhancement to suppress background noise while preserving the original recorded voice as much as possible.

---

## Overview

Real-world audio recordings often contain unwanted background noise, silence, music, environmental sounds, or other non-speech sections.

This project provides an automated audio-processing pipeline that focuses specifically on **human speech extraction and enhancement**.

The API:

1. Accepts an audio file.
2. Converts the input into a standard audio format using FFmpeg.
3. Detects human speech using Silero VAD.
4. Removes non-speech portions.
5. Enhances the detected speech using SpeechBrain.
6. Returns the processed speech as a WAV file.

The project is designed as a backend API and can be integrated with web applications, desktop applications, or other services.

---

## Key Features

- Human speech detection from noisy recordings
- Non-speech segment removal
- Background noise suppression
- Multiple audio format support
- Automatic audio format conversion
- FastAPI REST API
- Swagger/OpenAPI documentation
- CPU-based processing
- Original recorded speech is preserved rather than generating synthetic speech
- Temporary uploaded and converted files are automatically cleaned up

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
                  Standard WAV
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
                 Cleaned Speech
                         |
                         v
                   WAV Output
