import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path(__file__).resolve().parent / "lung-sound.wav"
y, fs = sf.read(file_path)

if y.ndim > 1:
    y = y[:, 0]

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración total: {len(y)/fs:.2f} s")

segment_duration = 2.0
N = int(segment_duration * fs)
hop = N // 2

best_score = -1
best_segment = None
best_start = 0

for start in range(0, len(y) - N, hop):
    segment = y[start:start+N]
    window = np.hanning(N)
    segment_w = segment * window

    Y = np.fft.rfft(segment_w)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(Y) * 2 / N

    band = (freqs >= 400) & (freqs <= 2000)
    score = np.max(mag[band])

    if score > best_score:
        best_score = score
        best_segment = segment
        best_start = start

window = np.hanning(N)
segment_w = best_segment * window

Y = np.fft.rfft(segment_w)
freqs = np.fft.rfftfreq(N, 1/fs)
mag = np.abs(Y) * 2 / N

band = (freqs >= 400) & (freqs <= 2000)
freq_dom = freqs[band][np.argmax(mag[band])]
amp_dom = mag[band][np.argmax(mag[band])]

start_time = best_start / fs
end_time = (best_start + N) / fs

print(f"\nSegmento analizado: {start_time:.2f} s a {end_time:.2f} s")
print(f"Frecuencia dominante detectada: {freq_dom:.2f} Hz")
print(f"Amplitud del pico dominante: {amp_dom:.4f}")

if freq_dom > 400:
    print("Conclusión: compatible con sibilancia.")
else:
    print("Conclusión: no parece una sibilancia típica.")

t_seg = np.linspace(start_time, end_time, N, endpoint=False)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t_seg, best_segment, color="blue")
plt.title("Segmento representativo en el dominio del tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqs, mag, color="red")
plt.title("Espectro de magnitud (FFT con ventana Hann)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.xlim(0, 2000)
plt.grid(True)

plt.tight_layout()
plt.show()