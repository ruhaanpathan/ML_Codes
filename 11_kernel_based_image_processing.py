# Kernel-based Image Processing using OpenCV
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a sample image so this program runs without an external image file.
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (250, 250), 180, -1)
cv2.circle(image, (150, 150), 55, 255, -1)
noise = np.random.default_rng(42).normal(0, 18, image.shape)
noisy_image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

# 3x3 averaging kernel
average_kernel = np.ones((3, 3), np.float32) / 9
average = cv2.filter2D(noisy_image, -1, average_kernel)

# Gaussian blur
gaussian = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

# Edge detection kernel (Laplacian)
edge_kernel = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float32)
edges = cv2.filter2D(image, cv2.CV_64F, edge_kernel)
edges = cv2.convertScaleAbs(edges)

images = [noisy_image, average, gaussian, sharpened, edges]
titles = ["Noisy Image", "Average Filter", "Gaussian Blur", "Sharpened", "Laplacian Edges"]

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
