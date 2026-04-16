import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal con ruido
x = np.sin(2 * np.pi * 60 * t) + 0.5 * np.random.randn(len(t))

N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Señal con ruido en el tiempo")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias de la señal con ruido")
plt.xlim(0, 150)
plt.grid(True)

plt.tight_layout()
plt.show()