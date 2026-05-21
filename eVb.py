import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 250   
t = np.arange(0, 2.0, 1/fs)   

ritmo_alfa = 1.2 * np.sin(2 * np.pi * 10 * t)  # Onda a 10 Hz (Dentro del rango 8-12 Hz)
ritmo_beta = 0.6 * np.sin(2 * np.pi * 22 * t)  # Onda a 22 Hz (Dentro del rango 12-30 Hz)
eeg_real = ritmo_alfa + ritmo_beta
ruido_ojos = 4.0 * np.sin(2 * np.pi * 1 * t)      # Parpadeo de ojos (Muy baja frecuencia: 1 Hz)
ruido_musculos = 2.5 * np.sin(2 * np.pi * 40 * t)  # Movimiento del cuello (Alta frecuencia: 40 Hz)
# Señal cruda que recibe el microprocesador del casco
eeg_ruidoso = eeg_real + ruido_ojos + ruido_musculos

orden_fir = 61
coeficientes = signal.firwin(orden_fir, [8, 30], fs=fs, window='hamming', pass_zero=False)
senal_filtrada = signal.lfilter(coeficientes, 1.0, eeg_ruidoso)

orden_iir = 8
b, a = signal.butter(orden_iir, [8, 30], fs=fs, btype='bandpass')
senal_filtrada2 = signal.lfilter(b, a, eeg_ruidoso)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, eeg_ruidoso, label='Señal EEG ruidosa (10Hz + 22Hz + 1Hz + 40Hz)', alpha=0.7)
plt.title('Señal original contaminada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, senal_filtrada, 'g', label='Salida del filtro FIR', linewidth=2)
plt.title('Señal filtrada (FIR pasa banda 8-30 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, senal_filtrada2, 'r', label='Salida del filtro IIR', linewidth=2)
plt.title('Señal filtrada (IIR Butterworth pasa banda 8-30 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

impulso = np.zeros(120)
impulso[0] = 1
respuesta_fir = signal.lfilter(coeficientes, 1.0, impulso)

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(coeficientes, color='blue')
plt.title("Coeficientes del filtro FIR")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(respuesta_fir, color='red')
plt.title("Respuesta al impulso del filtro FIR")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(t, eeg_real, color='black', linewidth=2, label='EEG real')
plt.plot(t, eeg_ruidoso, color='gray', alpha=0.4, label='EEG ruidoso')
plt.plot(t, senal_filtrada, color='green', label='FIR')
plt.plot(t, senal_filtrada2, color='red', label='IIR')
plt.title("Comparación directa de señales")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()
