import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import time
from google.colab import files
from PIL import Image
from pillow_heif import register_heif_opener
import time


uploaded = files.upload()  # Choose your HEIC/JPEG file

# Get the filename of the uploaded file
uploaded_filename = list(uploaded.keys())[0]

# Extract the file extension
file_extension = uploaded_filename.split(".")[-1].lower()

# Conditional conversion based on file type
if file_extension == "heic":
    register_heif_opener()  # Enable HEIF support
    heif_file = uploaded_filename
    image = Image.open(heif_file)
    converted_path = uploaded_filename.replace(".heic", ".png")
elif file_extension in ["jpg", "jpeg"]:
    jpeg_file = uploaded_filename
    image = Image.open(jpeg_file)
    converted_path = uploaded_filename.replace(f".{file_extension}", ".png")
else:
    print(f"Unsupported file type: {file_extension}")
    exit()  # Stop execution for unsupported types

image.save(converted_path, 'PNG')
print(f"Image converted and saved to: {converted_path}")


image = cv2.imread(converted_path, cv2.IMREAD_COLOR_RGB)

r, g, b = cv2.split(image)

identity_kernel = np.zeros((3,3), np.float32)
identity_kernel[1,1] = 1  # Set only the center pixel to 1


kernels = {
    "Blur": np.ones((25,25), np.float32) / 25**2,  # Blurs the image
    "Sharpen": np.array([
        [0, -1, 0],
        [-1, 8, -1],
        [0, -1, 0]]),  # Sharpens edges
    "Edge Detection": np.array([
        [-1, -1, -1],
        [-1, 0, -1],
        [-1, -1, -1]]),  # Detects edges
    "Emboss": np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]]),  # Gives an embossed look
    "Identity Kernal":  identity_kernel

}
start_time = time.time()
r_filtered = scipy.ndimage.convolve(r, kernels["Edge Detection"])
g_filtered = scipy.ndimage.convolve(g, kernels["Sharpen"])
b_filtered = scipy.ndimage.convolve(b, kernels["Sharpen"])


filtered_image = cv2.merge([r_filtered, g_filtered, b_filtered])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Convolution took {elapsed_time:.6f} seconds")

plt.figure(figsize=(10,13))
plt.imshow(filtered_image)
plt.axis("off")
plt.show()
