import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path, sr=16000):
    audio, sr = librosa.load(file_path, sr=sr)
    max_len = 5 * sr
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    else:
        audio = audio[:max_len]
    return audio, sr

def extract_acoustic_features(audio, sr):
    zcr = librosa.feature.zero_crossing_rate(audio).mean()
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_mean = np.mean(f0[f0 > 0]) if len(f0[f0 > 0]) > 0 else 0
    rms = librosa.feature.rms(y=audio).mean()
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean = np.mean(mel_spec)

    return {
        "zero_cross_rate": round(zcr, 4),
        "f0_mean": round(f0_mean, 2),
        "rms_energy": round(rms, 4),
        "mel_spec_mean": round(mel_mean, 4)
    }