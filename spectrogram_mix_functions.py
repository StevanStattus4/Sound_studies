# Spectrogram mixing:

#1) faz o mix de dois áudios no domínio do espectrograma com controle de SNR
#2) plotar três espectrogramas
#3) plotar três formas de onda e exibir os áudios

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import soundfile as sf


def mix_spectrograms(y_target, y_noise, sr, snr_db=10, n_fft=1024, hop_length=None):
    """Mix two signals in the STFT (magnitude) domain with control of SNR.

    Parameters
    ----------
    y_target : np.ndarray
        Signal principal (ex: vazamento).
    y_noise : np.ndarray
        Signal de ruído a ser adicionado (ex: latidos, pessoas, etc).
    sr : int
        Taxa de amostragem.
    snr_db : float
        SNR desejada em decibéis (signal power relative to noise power).
    n_fft : int
        Tamanho da janela FFT.
    hop_length : int or None
        Hop length para STFT. Se None, usa n_fft // 4.

    Returns
    -------
    y_mix : np.ndarray
        Sinal reconstruído (domínio do tempo) a partir da mistura espectral.
    S_mix_mag : np.ndarray
        Magnitude do espectrograma misto (frequência x tempo).
    alpha : float
        Fator aplicado ao espectrograma de ruído.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    # garante mesmo comprimento
    min_len = min(len(y_target), len(y_noise))
    y_target = y_target[:min_len]
    y_noise = y_noise[:min_len]

    # STFT 
    S_target = librosa.stft(y_target, n_fft=n_fft, hop_length=hop_length)
    S_noise = librosa.stft(y_noise, n_fft=n_fft, hop_length=hop_length)

    # Magnitudes
    M_target = np.abs(S_target)
    M_noise = np.abs(S_noise)

    # Potência média (magnitude^2)
    P_target = np.mean(M_target ** 2)
    P_noise = np.mean(M_noise ** 2)

    # Cálculo do alpha a partir da SNR desejada
    alpha = np.sqrt((P_target / (P_noise + 1e-12)) * 10 ** (-snr_db / 10))

    # Mistura das magnitudes
    M_mix = M_target + alpha * M_noise

    # Reaplica a fase do sinal alvo para construir a STFT mista
    phase_target = np.angle(S_target)
    S_mix = M_mix * np.exp(1j * phase_target)

    # Reconstrução (ISTFT)
    y_mix = librosa.istft(S_mix, hop_length=hop_length)

    # Normaliza 
    max_val = np.max(np.abs(y_mix))
    if max_val > 0:
        y_mix = y_mix / max_val

    return y_mix, M_mix, alpha


def plot_spectrograms(signals, sr, n_fft=1024, hop_length=None, titles=None, vmax_db=None):
    """Plota espectrogramas (em dB) de até três sinais.

    Parameters
    ----------
    signals : list of np.ndarray
        Lista de sinais (até 3) ou espectrogramas de magnitude. Se forem sinais, a função calcula STFT.
    sr : int
        Taxa de amostragem.
    n_fft, hop_length : int
        Parâmetros de STFT.
    titles : list of str
        Títulos para cada subplot.
    vmax_db : float or None
        Valor máximo de dB para normalização do colormap (opcional).
    """
    if hop_length is None:
        hop_length = n_fft // 4

    # Padroniza lista
    if titles is None:
        titles = [f"Sinal {i+1}" for i in range(len(signals))]

    plt.figure(figsize=(10, 3 * len(signals)))

    for i, s in enumerate(signals):
        # Detecta se já é uma magnitude de espectrograma (2D) ou um sinal 1D
        if isinstance(s, np.ndarray) and s.ndim == 1:
            S = np.abs(librosa.stft(s, n_fft=n_fft, hop_length=hop_length))
        else:
            S = s

        S_db = librosa.amplitude_to_db(S, ref=np.max)

        plt.subplot(len(signals), 1, i + 1)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title(titles[i])
        plt.colorbar(format='%+2.0f dB')
        if vmax_db is not None:
            img.set_clim(vmin=vmax_db[0], vmax=vmax_db[1])

    plt.tight_layout()
    plt.show()


def plot_waveforms_and_audio(signals, sr, titles=None, save_paths=None):
    """Plota as formas de onda dos sinais e exibe players de áudio (Jupyter).

    Parameters
    ----------
    signals : list of np.ndarray
        Lista de sinais 1D.
    sr : int
        Taxa de amostragem.
    titles : list of str
        Títulos para cada subplot.
    save_paths : list of str or None
        Se informado, salva cada sinal no caminho correspondente (WAV).
    """
    if titles is None:
        titles = [f"Sinal {i+1}" for i in range(len(signals))]

    # Salva se pedido
    if save_paths is not None:
        for s, p in zip(signals, save_paths):
            sf.write(p, s, sr)

    # Plot
    plt.figure(figsize=(12, 3 * len(signals)))
    t = np.arange(len(signals[0])) / sr

    for i, s in enumerate(signals):
        plt.subplot(len(signals), 1, i + 1)
        plt.plot(t[:len(s)], s)
        plt.title(titles[i])
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

    # Players de áudio
    for i, s in enumerate(signals):
        print(f"🔊 {titles[i]}")
        display(Audio(s, rate=sr))


# ===== Exemplo de uso =====
# import librosa
# y_vaz, sr = librosa.load('vazamento.wav', sr=None)
# y_ruido, _ = librosa.load('ruido.wav', sr=sr)
#
# y_mix, S_mix, alpha = mix_spectrograms(y_vaz, y_ruido, sr, snr_db=10)
# plot_spectrograms([y_vaz, y_ruido, S_mix], sr, titles=['Vazamento', 'Ruído', 'Mistura (mag)'])
# plot_waveforms_and_audio([y_vaz, y_ruido, y_mix], sr, titles=['Vazamento', 'Ruído', 'Mistura'])


