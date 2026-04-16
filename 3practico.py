import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Señal con dos frecuencias
x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)

# FFT de la señal original
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# Eliminar la frecuencia menor (50 Hz) en el dominio de la frecuencia
X_filtrada = X_fft.copy()

for i in range(len(frecuencias)):
    if abs(frecuencias[i]) == 50:
        X_filtrada[i] = 0

# Señal reconstruida sin la frecuencia de 50 Hz
x_filtrada = np.fft.ifft(X_filtrada).real
magnitud_filtrada = np.abs(X_filtrada[:n_mitad]) * 2 / N

# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, x)
plt.title("Señal original en el tiempo")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias de la señal original")
plt.xlim(0, 250)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(f_positivas, magnitud_filtrada)
plt.title("Espectro después de eliminar la frecuencia de 50 Hz")
plt.xlim(0, 250)
plt.grid(True)

plt.tight_layout()
plt.show()