import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

file_path = Path(__file__).resolve().parent / "lung-sound.wav"
y, fs = sf.read(file_path)

if y.ndim > 1:
    y = y[:, 0]

freq_sibilancia = 464.3   

margen = 80.0             

lowcut = freq_sibilancia - margen
highcut = freq_sibilancia + margen

if lowcut <= 0:
    lowcut = 1.0
if highcut >= fs / 2:
    highcut = fs / 2 - 1

def butter_bandpass(lowcut, highcut, fs, order=4):
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return sos

sos = butter_bandpass(lowcut, highcut, fs, order=4)

y_filtrado = sosfiltfilt(sos, y)

max_val = np.max(np.abs(y_filtrado))
if max_val > 0:
    y_filtrado = y_filtrado / max_val

output_path = Path(__file__).resolve().parent / "sibilancia_aislada.wav"
sf.write(output_path, y_filtrado, fs)

print(f"Frecuencia central aislada: {freq_sibilancia:.2f} Hz")
print(f"Banda del filtro: {lowcut:.2f} Hz a {highcut:.2f} Hz")
print(f"Archivo guardado como: {output_path.name}")

N = len(y)
Y = np.fft.rfft(y)
Yf = np.fft.rfft(y_filtrado)

freqs = np.fft.rfftfreq(N, 1/fs)
mag_original = np.abs(Y) * 2 / N
mag_filtrado = np.abs(Yf) * 2 / N

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(freqs, mag_original, color="gray")
plt.title("Espectro original")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqs, mag_filtrado, color="blue")
plt.title("Espectro de la sibilancia aislada")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.tight_layout()
plt.show()