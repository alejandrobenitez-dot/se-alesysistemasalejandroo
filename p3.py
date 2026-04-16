import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# Cargar audio
y, fs = librosa.load('audioJB.wav', sr=None)

# FFT
N = len(y)
Y_fft = np.fft.fft(y)
frecuencias = np.fft.fftfreq(N, 1/fs)

# Filtro pasa-bajo
fc = 1000  # frecuencia de corte en Hz
Y_filtrado = Y_fft.copy()
Y_filtrado[np.abs(frecuencias) > fc] = 0

# Reconstruir señal filtrada
y_filtrado = np.fft.ifft(Y_filtrado).real

# Guardar audio filtrado
sf.write('audio_filtrado.wav', y_filtrado, fs)

# Solo frecuencias positivas
n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud_original = np.abs(Y_fft[:n_mitad])
magnitud_filtrada = np.abs(Y_filtrado[:n_mitad])

# Frecuencia dominante después del filtrado
mask = f_positivas < fc
indice_max = np.argmax(magnitud_filtrada[mask])
frecuencia_dominante = f_positivas[mask][indice_max]

# Gráficas
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(y)
plt.title("Señal de audio original")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(f_positivas, magnitud_original)
plt.title("Espectro del audio original")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 5000)
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(f_positivas, magnitud_filtrada)
plt.title("Espectro del audio filtrado con pasa-bajo")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 5000)
plt.grid(True)

plt.tight_layout()
plt.show()

print("Frecuencia dominante después del filtrado:", round(frecuencia_dominante, 2), "Hz")