import numpy as np
import matplotlib.pyplot as plt
import pickle

# noise_path="/home/yichenyu/git/robust-unlearnable-examples/exp_data/voc/lsp-fin-eps8-def-noise.pkl"
noise_path="fourier_basis_sample/fourier_basis224.npy"

with open(noise_path, 'rb') as f:
    images = np.load(f)

image=images[4,4]
f_transform = np.fft.fft2(image.squeeze())
f_magnitude = np.abs(f_transform)

# Visualize the result and save to a PNG file
plt.imshow(np.fft.fftshift(f_magnitude), cmap='gray')
plt.title("Average Magnitude of Fourier Coefficients")
plt.colorbar()
plt.savefig('average_fourier_sps.png')  # Save the image as a PNG file
plt.close()  # Close the plot to free up memory
