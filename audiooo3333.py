import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# --- CONFIGURACIÓN DE LA PRÁCTICA ---
FS = 44100          # Frecuencia de muestreo (Hz)
N = 4096            # Tamaño de la ventana (N)
OVERLAP = 0.5       # 50% de solapamiento
HOP_SIZE = int(N * (1 - OVERLAP)) # Cada cuántas muestras procesamos

# Definir la función de ventana
window = np.hanning(N)

# Buffer para acumular muestras suficientes para la ventana N
audio_buffer = np.zeros(N)

# Configuración de la gráfica
fig, ax = plt.subplots()
x_freq = rfftfreq(N, 1/FS)
line, = ax.plot(x_freq, np.zeros(len(x_freq)))
ax.set_ylim(0, 50) # Ajustar según la sensibilidad del micro
ax.set_xlim(0, 5000) 
ax.set_xlabel('Frecuencia (Hz)')
ax.set_ylabel('Magnitud (dB)')

def audio_callback(indata, frames, time, status):
    """Esta función se llama cada vez que la tarjeta de sonido tiene datos."""
    global audio_buffer
    new_samples = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -len(new_samples))
    audio_buffer[-len(new_samples):] = new_samples

    # 1. Aplicar Windowing
    windowed_data = audio_buffer * window
    
    # 2. Calcular FFT
    yf = rfft(windowed_data)
    mag = 20 * np.log10(np.abs(yf) + 1e-9) # Convertir a dB
    
    # 3. Actualizar la línea de la gráfica
    line.set_ydata(mag)

# Iniciar el flujo de audio
try:
    with sd.InputStream(channels=1, callback=audio_callback, 
                        blocksize=HOP_SIZE, samplerate=FS):
        print("Grabando... Cierra la ventana para detener.")
        while plt.fignum_exists(fig.number):
            plt.pause(0.01)
except Exception as e:
    print(f"Error: {e}")