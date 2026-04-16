import numpy as np
import matplotlib.pyplot as plt

# Parámetros
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal de 40 Hz
x = np.sin(2 * np.pi * 40 * t)

# FFT
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

# Solo frecuencias positivas
n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# Gráficas
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, x)
plt.title("Señal en el tiempo (40 Hz)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias")
plt.xlim(0, 100)
plt.grid(True)

plt.tight_layout()
plt.show()