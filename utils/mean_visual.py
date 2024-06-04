import numpy as np
import matplotlib.pyplot as plt
import pickle

def calculate_and_save_mean_image(images, num_samples=1, output_filename='average_image_em.png'):
    # Randomly sample 1000 images
    sampled_indices = np.random.choice(images.shape[0], num_samples, replace=False)
    sampled_images = images[sampled_indices]

    # Calculate the average image
    average_image = np.mean(sampled_images, axis=0)

    # Visualize the result and save to a PNG file
    plt.imshow(average_image.squeeze())
    # plt.title("Average Image")
    plt.colorbar()
    plt.savefig(output_filename)  # Save the image as a PNG file
    plt.close()  # Close the plot to free up memory

# Example usage:
# Assume images is a 4D numpy array with shape (num_images, height, width, channels)
# For grayscale images, channels should be 1. For RGB images, channels should be 3.
# images = np.random.rand(5000, 64, 64, 1)  # Example array, replace with your actual array
# noise_path="../runs/train/yolov7-tap07124/TAP-def-noise.pkl"
noise_path="../runs/train/yolov7-em07123/EM-def-noise.pkl"
# noise_path="/home/yichenyu/git/robust-unlearnable-examples/exp_data/voc/lsp-fin-eps8-def-noise.pkl"

with open(noise_path, 'rb') as f:
    images = pickle.load(f)
    images=np.transpose(images,(0,2,3,1))*20
    images=np.clip(images,0,1)

print("noise loaded")
calculate_and_save_mean_image(images)
