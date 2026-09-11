from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from speechbrain.inference.enhancement import WaveformEnhancement
from speechbrain.utils.fetching import LocalStrategy
from silero_vad import load_silero_vad, get_speech_timestamps



# CONFIGURATION


MODEL_SAMPLE_RATE = 16000

DEVICE = "cpu"




VAD_THRESHOLD = 0.45

MIN_SPEECH_DURATION_MS = 300

MIN_SILENCE_DURATION_MS = 500

SPEECH_PAD_MS = 150



# OUTPUT SETTINGS


TARGET_PEAK = 0.95



# PROJECT DIRECTORIES


BASE_DIR = Path(__file__).resolve().parent.parent

PRETRAINED_DIR = (
    BASE_DIR
    / "pretrained_models"
    / "speechbrain_enhancement"
)

PRETRAINED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)



# LOAD SPEECHBRAIN ENHANCEMENT MODEL


print()

print("Loading SpeechBrain speech enhancement model...")


print(
    "Model directory:",
    PRETRAINED_DIR,
)

ENHANCER = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",

    savedir=str(
        PRETRAINED_DIR
    ),

    run_opts={
        "device": DEVICE,
    },

   
    # Windows fix:
    # COPY avoids symbolic-link permission problems.
    

    local_strategy=LocalStrategy.COPY,
)

print(
    "SpeechBrain enhancement model loaded."
)


# LOAD SILERO VAD


print()

print("Loading Silero VAD model...")


VAD_MODEL = load_silero_vad(
    onnx=False
)

print(
    "Silero VAD loaded."
)



# LOAD AUDIO


def load_audio(input_file):
   
    audio, sample_rate = sf.read(
        input_file,
        always_2d=False,
        dtype="float32",
    )

    print()
    print(
        "Processing WAV sample rate:",
        sample_rate,
    )

    print(
        "Original audio shape:",
        audio.shape,
    )

   
    # Stereo / multi-channel -> mono
   

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
        )

    
    # Ensure float32
    

    audio = audio.astype(
        np.float32
    )

    
    # Remove DC offset
    

    audio = (
        audio
        - np.mean(audio)
    )

    return audio, sample_rate



# RESAMPLE AUDIO

def resample_audio(
    audio,
    original_sample_rate,
    target_sample_rate,
):
    

    if (
        original_sample_rate
        == target_sample_rate
    ):

        return audio.astype(
            np.float32
        )

    print(
        f"Resampling "
        f"{original_sample_rate} Hz -> "
        f"{target_sample_rate} Hz"
    )

    waveform = torch.from_numpy(
        audio
    ).float()

    # [samples] -> [1, samples]

    waveform = waveform.unsqueeze(0)

    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )

    # [1, samples] -> [samples]

    waveform = waveform.squeeze(0)

    return (
        waveform
        .numpy()
        .astype(np.float32)
    )



# NORMALIZE AUDIO


def normalize_audio(audio):
    """
    Safely normalize audio.

    Only reduces the level when necessary.
    It does not aggressively amplify very quiet recordings.
    """

    if len(audio) == 0:

        return audio.astype(
            np.float32
        )

    peak = np.max(
        np.abs(audio)
    )

    if peak < 1e-8:

        return audio.astype(
            np.float32
        )

   
    # Only reduce audio if it exceeds target peak.
   

    if peak > TARGET_PEAK:

        audio = (
            audio
            / peak
            * TARGET_PEAK
        )

    return audio.astype(
        np.float32
    )



# DETECT HUMAN SPEECH


def detect_speech(audio_16k):
    """
    Detect speech using Silero VAD.

    The output contains the start/end sample positions
    of detected speech regions.
    """

    print()
    print("=" * 60)
    print("DETECTING HUMAN SPEECH")
    print("=" * 60)

    if len(audio_16k) == 0:

        raise ValueError(
            "Audio is empty."
        )

    waveform = torch.from_numpy(
        audio_16k
    ).float()


    # Silero VAD
   

    speech_timestamps = get_speech_timestamps(
        waveform,

        VAD_MODEL,

        sampling_rate=MODEL_SAMPLE_RATE,

        # Speech detection sensitivity
        threshold=VAD_THRESHOLD,

        # Ignore extremely short sounds
        min_speech_duration_ms=(
            MIN_SPEECH_DURATION_MS
        ),

        # Join speech separated by short silence
        min_silence_duration_ms=(
            MIN_SILENCE_DURATION_MS
        ),

        # Preserve a small amount around speech boundaries
        speech_pad_ms=SPEECH_PAD_MS,

        return_seconds=False,
    )

    print(
        "Speech regions detected:",
        len(speech_timestamps),
    )

  
    # Print detected regions
 
    for index, segment in enumerate(
        speech_timestamps,
        start=1,
    ):

        start_seconds = (
            segment["start"]
            / MODEL_SAMPLE_RATE
        )

        end_seconds = (
            segment["end"]
            / MODEL_SAMPLE_RATE
        )

        print(
            f"Speech {index}: "
            f"{start_seconds:.2f}s -> "
            f"{end_seconds:.2f}s"
        )

    return speech_timestamps



# EXTRACT ONLY SPEECH


def extract_speech_regions(
    audio,
    speech_timestamps,
):
    

    if not speech_timestamps:

        raise ValueError(
            "No human speech was detected in the audio."
        )

    speech_parts = []

    for segment in speech_timestamps:

        start = int(
            segment["start"]
        )

        end = int(
            segment["end"]
        )

        if end <= start:

            continue

        speech_part = audio[
            start:end
        ]

        if len(speech_part) == 0:

            continue

        speech_parts.append(
            speech_part
        )

    if not speech_parts:

        raise ValueError(
            "No valid speech segments were detected."
        )

    # Join all speech regions.
   
    # This removes background-only sections completely.
   

    extracted = np.concatenate(
        speech_parts
    )

    return extracted.astype(
        np.float32
    )



# ENHANCE SPEECH


def enhance_speech(audio_16k):
    """
    Use SpeechBrain to suppress background noise
    while retaining the original speech signal.
    """

    print()
   
    print("SUPPRESSING BACKGROUND NOISE")
    

    if len(audio_16k) == 0:

        raise ValueError(
            "Cannot enhance empty audio."
        )

    waveform = torch.from_numpy(
        audio_16k
    ).float()

    
    # SpeechBrain expects:
    
    if waveform.ndim == 1:

        waveform = waveform.unsqueeze(0)

    print(
        "Enhancing speech..."
    )

    with torch.no_grad():

        enhanced = ENHANCER.enhance_batch(
            waveform
        )

    
    # Some SpeechBrain versions may return a tuple.
    

    if isinstance(
        enhanced,
        tuple,
    ):

        enhanced = enhanced[0]

   
    # Convert to NumPy
   
    enhanced = (
        enhanced
        .detach()
        .cpu()
        .numpy()
    )

    
    # Remove unnecessary dimensions
    

    enhanced = np.squeeze(
        enhanced
    )

    
    # Safety check
    

    if enhanced.ndim != 1:

        enhanced = enhanced.reshape(
            -1
        )

    return enhanced.astype(
        np.float32
    )



# MAIN PROCESSOR


def process_audio(
    input_file,
    output_file,
):
  

    print()
    
    print("HUMAN SPEECH EXTRACTION")
   

    input_file = str(
        Path(input_file)
    )

    output_file = str(
        Path(output_file)
    )

    
    # 1. LOAD AUDIO
    

    audio, original_sample_rate = load_audio(
        input_file
    )

    if len(audio) == 0:

        raise ValueError(
            "Input audio is empty."
        )

    original_duration = (
        len(audio)
        / original_sample_rate
    )

    print(
        f"Original duration: "
        f"{original_duration:.2f} seconds"
    )

    
    # 2. RESAMPLE TO 16 kHz
    

    audio_16k = resample_audio(
        audio,
        original_sample_rate,
        MODEL_SAMPLE_RATE,
    )

   
    # 3. DETECT HUMAN SPEECH
   

    speech_timestamps = detect_speech(
        audio_16k
    )

    
    # 4. EXTRACT ONLY HUMAN SPEECH
    

    extracted_speech = extract_speech_regions(
        audio_16k,
        speech_timestamps,
    )

    extracted_duration = (
        len(extracted_speech)
        / MODEL_SAMPLE_RATE
    )

    print()
    print(
        f"Speech-only duration: "
        f"{extracted_duration:.2f} seconds"
    )

    removed_duration = max(
        0,
        original_duration - extracted_duration,
    )

    print(
        f"Removed approximately: "
        f"{removed_duration:.2f} seconds"
    )

  
    # 5. SPEECH ENHANCEMENT
    

    enhanced_speech = enhance_speech(
        extracted_speech
    )

    
    # 6. RESTORE ORIGINAL SAMPLE RATE
  
    if (
        original_sample_rate
        != MODEL_SAMPLE_RATE
    ):

        enhanced_speech = resample_audio(
            enhanced_speech,
            MODEL_SAMPLE_RATE,
            original_sample_rate,
        )

        output_sample_rate = (
            original_sample_rate
        )

    else:

        output_sample_rate = (
            MODEL_SAMPLE_RATE
        )

   
    # 7. NORMALIZE
   

    enhanced_speech = normalize_audio(
        enhanced_speech
    )

 
    # 8. CREATE OUTPUT DIRECTORY
    
    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

   
    # 9. SAVE OUTPUT
    

    print()
    print(
        "Saving extracted human speech..."
    )

    sf.write(
        str(output_path),
        enhanced_speech,
        output_sample_rate,
        subtype="PCM_16",
    )

  
    # 10. VERIFY OUTPUT
   
    if not output_path.exists():

        raise RuntimeError(
            "Output WAV file was not created."
        )

    if output_path.stat().st_size == 0:

        raise RuntimeError(
            "Output WAV file is empty."
        )

   
    # 11. FINAL INFORMATION
    

    final_duration = (
        len(enhanced_speech)
        / output_sample_rate
    )

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(
        "Input:",
        input_file,
    )

    print(
        "Output:",
        output_file,
    )

    print(
        f"Original duration: "
        f"{original_duration:.2f}s"
    )

    print(
        f"Speech duration: "
        f"{final_duration:.2f}s"
    )

    print(
        f"Removed approximately: "
        f"{max(0, original_duration - final_duration):.2f}s"
    )

    print(
        "Output sample rate:",
        output_sample_rate,
    )

    print(
        "Output size:",
        output_path.stat().st_size,
        "bytes",
    )

   

    return str(
        output_path
    )