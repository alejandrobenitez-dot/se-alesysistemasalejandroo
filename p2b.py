import numpy as np
import librosa
import matplotlib.pyplot as plt

# Cargar audio
y, fs = librosa.load('audioJB.wav', sr=None)

# Generar ruido
ruido = 0.05 * np.random.randn(len(y))

# Señal con ruido
y_ruido = y + ruido

# FFT del audio original
N = len(y)
Y_fft = np.fft.fft(y)
frecuencias = np.fft.fftfreq(N, 1/fs)

# FFT del audio con ruido
Y_ruido_fft = np.fft.fft(y_ruido)

# Solo frecuencias positivas
n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud_original = np.abs(Y_fft[:n_mitad])
magnitud_ruido = np.abs(Y_ruido_fft[:n_mitad])

# Graficas
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.plot(y_ruido)
plt.title("Señal de audio con ruido")
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(f_positivas, magnitud_original)
plt.title("Espectro del audio original")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 5000)
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(f_positivas, magnitud_ruido)
plt.title("Espectro del audio con ruido")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(0, 5000)
plt.grid(True)

plt.tight_layout()
plt.show()