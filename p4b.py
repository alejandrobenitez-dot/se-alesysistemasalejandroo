import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
img = cv2.imread('imagen.jpg', 0)

# FFT bidimensional
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Espectro de amplitud
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Crear máscara pasa-bajo
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

mask = np.zeros((rows, cols), np.uint8)
r = 40
cv2.circle(mask, (ccol, crow), r, 1, -1)

# Aplicar filtro
fshift_filtrado = fshift * mask

# Transformada inversa
f_ishift = np.fft.ifftshift(fshift_filtrado)
img_filtrada = np.fft.ifft2(f_ishift)
img_filtrada = np.abs(img_filtrada)

# Gráficas
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Espectro de amplitud")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(mask, cmap='gray')
plt.title("Máscara pasa-bajo")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_filtrada, cmap='gray')
plt.title("Imagen filtrada")
plt.axis('off')

plt.tight_layout()
plt.show()