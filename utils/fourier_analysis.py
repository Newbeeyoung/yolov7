import numpy as np
import matplotlib.pyplot as plt
import pickle

# noise_path="/home/yichenyu/git/robust-unlearnable-examples/exp_data/voc/lsp-fin-eps8-def-noise.pkl"
noise_path="../runs/train/yolov7-tap07124/TAP-def-noise.pkl"
# noise_path="../runs/train/yolov7-em07123/EM-def-noise.pkl"

with open(noise_path, 'rb') as f:
    images = pickle.load(f)
    images=np.transpose(images,(0,2,3,1))
# Assume images is a 4D numpy array with shape (num_images, height, width, channels)
# For grayscale images, channels should be 1. For RGB images, channels should be 3.
# images = np.random.rand(5000, 64, 64, 1)  # Example array, replace with your actual array

# Parameters
num_samples = 1000

# Randomly sample 1000 images
sampled_indices = np.random.choice(images.shape[0], num_samples, replace=False)
sampled_images = images[sampled_indices]

# Initialize array to accumulate the magnitudes of the Fourier coefficients
magnitude_sum = np.zeros_like(sampled_images[0], dtype=np.float64)
print("Calculating Spectrum")
# Perform Fourier Transform and accumulate magnitudes
for image in sampled_images:
    f_transform = np.fft.fft2(image.squeeze())
    f_magnitude = np.abs(f_transform)
    magnitude_sum += f_magnitude

# Calculate the average magnitude
average_magnitude = magnitude_sum / num_samples

# Visualize the result and save to a PNG file
plt.imshow(np.fft.fftshift(average_magnitude), cmap='gray')
plt.title("Average Magnitude of Fourier Coefficients")
plt.colorbar()
plt.savefig('average_fourier_tap.png')  # Save the image as a PNG file
plt.close()  # Close the plot to free up memory
