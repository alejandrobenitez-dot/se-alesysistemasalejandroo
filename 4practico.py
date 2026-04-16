import numpy as np
import matplotlib.pyplot as plt

# Caso 2: frecuencia de muestreo alta
fs = 3000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal con dos frecuencias
x = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

# FFT
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, x)
plt.title("Señal en el tiempo con Fs = 3000 Hz")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias con Fs = 3000 Hz")
plt.xlim(0, 1500)
plt.grid(True)

plt.tight_layout()
plt.show()