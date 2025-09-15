# Pipeline  de extração de features de áudio 
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path
import numpy as np
import librosa


# -----------------------------
# Config
# -----------------------------
@dataclass
class FeatureConfig:
    # carga / pré-processamento
    sr: int = 16000
    mono: bool = True
    offset: float = 0.0
    duration: Optional[float] = None
    trim_silence: bool = True
    trim_top_db: float = 30.0
    normalize_peak: bool = True
    pre_emphasis: bool = True
    pre_emph_coef: float = 0.97

    # STFT
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None
    window: str = "hann"

    # Mel / MFCC
    n_mels: int = 64
    fmin: float = 20.0
    fmax: Optional[float] = None
    n_mfcc: int = 13
    mfcc_use_deltas: bool = True     # adiciona Δ e ΔΔ

    # Seleção de features
    use_logmel: bool = True
    use_mfcc: bool = True
    use_chroma: bool = True
    use_contrast: bool = True
    use_tonnetz: bool = False        # útil para sinais tonais
    use_rms: bool = True
    use_zcr: bool = True
    use_centroid_bw_rolloff: bool = True
    use_flatness: bool = True
    use_tempo: bool = True

    # Agregação
    aggregate: bool = True
    aggregate_stats: Tuple[str, ...] = ("mean", "std", "median", "iqr", "min", "max")

    # Segurança numérica
    eps: float = 1e-10


# -----------------------------
# Funções utilitárias
# -----------------------------
def _pre_emphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    if y.size < 2:
        return y
    y_out = np.empty_like(y)
    y_out[0] = y[0]
    y_out[1:] = y[1:] - coef * y[:-1]
    return y_out

def _iqr(x: np.ndarray, axis: int = -1) -> np.ndarray:
    q75 = np.nanpercentile(x, 75, axis=axis)
    q25 = np.nanpercentile(x, 25, axis=axis)
    return q75 - q25

def _aggregate_matrix(F: np.ndarray, name: str, stats: Tuple[str, ...]) -> Tuple[np.ndarray, List[str]]:
    """Aggrega (D, T) -> concat de stats por eixo 1 (tempo)."""
    F = np.atleast_2d(F)
    vec_parts, names = [], []
    for stat in stats:
        if stat == "mean":
            vals = np.nanmean(F, axis=1)
        elif stat == "std":
            vals = np.nanstd(F, axis=1)
        elif stat == "median":
            vals = np.nanmedian(F, axis=1)
        elif stat == "iqr":
            vals = _iqr(F, axis=1)
        elif stat == "min":
            vals = np.nanmin(F, axis=1)
        elif stat == "max":
            vals = np.nanmax(F, axis=1)
        else:
            raise ValueError(f"Estatística '{stat}' não suportada.")
        vec_parts.append(vals)
        names.extend([f"{name}_{i}_{stat}" for i in range(F.shape[0])])
    return np.concatenate(vec_parts, axis=0).astype(np.float32), names


# -----------------------------
# Extrator 
# -----------------------------
def extract_audio_features_advanced(
    y_or_path: Union[str, np.ndarray],
    cfg: FeatureConfig = FeatureConfig(),
    return_maps: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """
    Retorna:
      - features_dict: mapas 2D (n_feats x T) por chave
      - feature_vector: vetor 1D com agregações
      - feature_names: nomes alinhados ao vetor
    """
    # 1) Carregar áudio
    if isinstance(y_or_path, str):
        y, sr_loaded = librosa.load(
            y_or_path, sr=cfg.sr, mono=cfg.mono, offset=cfg.offset, duration=cfg.duration
        )
    else:
        y = np.asarray(y_or_path, dtype=np.float32)
        sr_loaded = cfg.sr if cfg.sr is not None else 22050
        if cfg.mono and y.ndim > 1:
            y = librosa.to_mono(y)

    # 2) Trim e normalização
    if cfg.trim_silence:
        y, _ = librosa.effects.trim(y, top_db=cfg.trim_top_db)
    if cfg.normalize_peak and np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + cfg.eps)
    if cfg.pre_emphasis:
        y = _pre_emphasis(y, coef=cfg.pre_emph_coef)

    # 3) STFT
    win_length = cfg.win_length or cfg.n_fft
    S_complex = librosa.stft(
        y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=win_length, window=cfg.window
    )
    S_mag = np.abs(S_complex)
    S_pow = (S_mag ** 2)
    features_dict: Dict[str, np.ndarray] = {}

    # 4) Log-Mel
    if cfg.use_logmel:
        mel = librosa.feature.melspectrogram(
            S=S_pow, sr=sr_loaded, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
        )
        logmel = np.log(mel + cfg.eps)
        features_dict["logmel"] = logmel

    # 5) MFCC 
    if cfg.use_mfcc:
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr_loaded, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length
        )
        if cfg.mfcc_use_deltas:
            d1 = librosa.feature.delta(mfcc, order=1)
            d2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.vstack([mfcc, d1, d2])  # shape: (n_mfcc*3, T)
        features_dict["mfcc"] = mfcc

    # 6) Chroma / Contrast / Tonnetz
    if cfg.use_chroma:
        features_dict["chroma"] = librosa.feature.chroma_stft(S=S_pow, sr=sr_loaded, hop_length=cfg.hop_length)
    if cfg.use_contrast:
        features_dict["contrast"] = librosa.feature.spectral_contrast(S=S_pow, sr=sr_loaded, fmin=cfg.fmin)
    if cfg.use_tonnetz:
        y_h = librosa.effects.harmonic(y)
        features_dict["tonnetz"] = librosa.feature.tonnetz(y=y_h, sr=sr_loaded)

    # 7) Outros espectrais
    if cfg.use_rms:
        features_dict["rms"] = librosa.feature.rms(S=S_mag, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
    if cfg.use_zcr:
        features_dict["zcr"] = librosa.feature.zero_crossing_rate(y, frame_length=win_length, hop_length=cfg.hop_length)
    if cfg.use_centroid_bw_rolloff:
        features_dict["centroid"]  = librosa.feature.spectral_centroid(S=S_pow, sr=sr_loaded)
        features_dict["bandwidth"] = librosa.feature.spectral_bandwidth(S=S_pow, sr=sr_loaded)
        features_dict["rolloff"]   = librosa.feature.spectral_rolloff(S=S_pow, sr=sr_loaded)
    if cfg.use_flatness:
        features_dict["flatness"] = librosa.feature.spectral_flatness(S=S_pow)

    # 8) Tempo (BPM)
    if cfg.use_tempo:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr_loaded, hop_length=cfg.hop_length)
            tempo = float(tempo)
        except Exception:
            tempo = np.nan
        T = next(iter(features_dict.values())).shape[1] if features_dict else S_mag.shape[1]
        features_dict["tempo"] = np.full((1, T), tempo, dtype=np.float32)

    # 9) Agregação -> vetor fixo
    feat_vec_list, feat_names = [], []
    if cfg.aggregate and features_dict:
        for k, v in features_dict.items():
            vec_k, names_k = _aggregate_matrix(v, k, cfg.aggregate_stats)
            feat_vec_list.append(vec_k)
            feat_names.extend(names_k)
        feature_vector = np.concatenate(feat_vec_list, axis=0).astype(np.float32)
    else:
        feature_vector = np.array([], dtype=np.float32)
        feat_names = []

    # 10) Retorno
    if not return_maps:
        # Se não quiser mapas, só retorna dict vazio + vetor
        return {}, feature_vector, feat_names
    return features_dict, feature_vector, feat_names


# -----------------------------
# Transformar em DataFrame + padroniza
# -----------------------------
def features_to_dataframe(feature_vector: np.ndarray, feature_names: List[str], tag: str = "sample_0"):
    if pd is None:
        raise ImportError("pandas não disponível. Instale `pandas` para usar este helper.")
    df = pd.DataFrame([feature_vector], index=[tag], columns=feature_names)
    return df

def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    if StandardScaler is None:
        print("⚠️ sklearn não disponível. Retornando X sem padronizar.")
        return X, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

print("✅ Pipeline avançado pronto: use `extract_audio_features_advanced(path_or_wave, cfg)`")


# === Exemplo: usando o pipeline ===

from pathlib import Path
from IPython.display import Audio, display

audio_path = "/home/stevan/Documentos/4fluid-ia/data/audios/2025_db1/3026c4d9-78ff-4dde-8034-fb2c18d5100f.wav"
cfg = FeatureConfig(
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    n_mfcc=13,
    mfcc_use_deltas=True,      # MFCC 
    use_logmel=True,
    use_mfcc=True,
    use_chroma=True,
    use_contrast=True,
    use_tonnetz=False,         # ative se o sinal for musical
    use_rms=True,
    use_zcr=True,
    use_centroid_bw_rolloff=True,
    use_flatness=True,
    use_tempo=True,
    aggregate=True,
    aggregate_stats=("mean","std","median","iqr","min","max"),
)

maps, vec, names = extract_audio_features_advanced(audio_path, cfg, return_maps=True)
print(f"✅ vetor agregado: {vec.shape} dimensões")
print("alguns nomes:", names[:10])
print("mapas disponíveis:", {k: v.shape for k, v in maps.items()})

if Path(audio_path).exists():
    display(Audio(audio_path))

