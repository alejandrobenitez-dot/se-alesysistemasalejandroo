import numpy as np
import librosa
import matplotlib.pyplot as plt

# Cargar audio
y, fs = librosa.load('audioJB.wav', sr=None)

# FFT
N = len(y)
Y_fft = np.fft.fft(y)
frecuencias = np.fft.fftfreq(N, 1/fs)

# Solo frecuencias positivas
n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(Y_fft[:n_mitad])

# Encontrar frecuencia dominante
mask = f_positivas < 2000
indice_max = np.argmax(magnitud[mask])
frecuencia_dominante = f_positivas[mask][indice_max]

# Graficas
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(y)
plt.title("Señal de audio en el tiempo")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 5000)
plt.grid(True)

plt.tight_layout()
plt.show()

print("Frecuencia dominante exacta:", round(frecuencia_dominante, 2), "Hz")