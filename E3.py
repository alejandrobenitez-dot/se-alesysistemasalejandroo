import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path(__file__).resolve().parent / "lung-sound.wav"
y, fs = sf.read(file_path)

if y.ndim > 1:
    y = y[:, 0]

freq_sibilancia = 464.3   
margen = 80.0             

lowcut = freq_sibilancia - margen
highcut = freq_sibilancia + margen

Y = np.fft.fft(y)
N = len(y)
freqs = np.fft.fftfreq(N, 1/fs)

Y_filtrado = Y.copy()

mask = (np.abs(freqs) >= lowcut) & (np.abs(freqs) <= highcut)
Y_filtrado[mask] = 0

y_limpio = np.fft.ifft(Y_filtrado)
y_limpio = np.real(y_limpio)


max_val = np.max(np.abs(y_limpio))
if max_val > 0:
    y_limpio = y_limpio / max_val

output_path = Path(__file__).resolve().parent / "lung_sound_limpio.wav"
sf.write(output_path, y_limpio, fs)

print(f"Frecuencia eliminada: {freq_sibilancia:.2f} Hz")
print(f"Banda eliminada: {lowcut:.2f} Hz a {highcut:.2f} Hz")
print(f"Archivo guardado como: {output_path.name}")

freqs_pos = np.fft.rfftfreq(N, 1/fs)
mag_original = np.abs(np.fft.rfft(y)) * 2 / N
mag_limpio = np.abs(np.fft.rfft(y_limpio)) * 2 / N

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(freqs_pos, mag_original, color="red")
plt.title("Espectro original")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqs_pos, mag_limpio, color="green")
plt.title("Espectro después de eliminar la sibilancia")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.tight_layout()
plt.show()