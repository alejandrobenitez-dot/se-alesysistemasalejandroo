import numpy as np
import matplotlib.pyplot as plt

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Frecuencia desconocida aleatoria entre 50 y 200 Hz
f_desconocida = np.random.randint(50, 201)

# Señal
x = np.sin(2 * np.pi * f_desconocida * t)

# FFT
N = len(x)
X_fft = np.fft.fft(x)
frecuencias = np.fft.fftfreq(N, 1/fs)

n_mitad = N // 2
f_positivas = frecuencias[:n_mitad]
magnitud = np.abs(X_fft[:n_mitad]) * 2 / N

# Identificar frecuencia dominante
indice_max = np.argmax(magnitud)
frecuencia_dominante = f_positivas[indice_max]

# Gráficas
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.plot(t, x)
plt.title("Señal en el tiempo con frecuencia desconocida")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(f_positivas, magnitud)
plt.title("Espectro de frecuencias")
plt.xlim(0, 250)
plt.grid(True)

plt.tight_layout()
plt.show()

print("Frecuencia generada aleatoriamente:", f_desconocida, "Hz")
print("Frecuencia dominante identificada con FFT:", frecuencia_dominante, "Hz")