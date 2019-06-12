
# coding: utf-8

# In[1]:


#Dependencies
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math
import PIL.Image
import inception5h #Thank you Hvass
from scipy.ndimage.filters import gaussian_filter
from IPython.display import Image, display


inception5h.maybe_download()
model = inception5h.Inception5h()


def load_image(filename):
    image = PIL.Image.open(filename)
    return np.float32(image)
def save_image(image, filename):    
    image = np.clip(image, 0.0, 255.0)    
    image = image.astype(np.uint8)    
    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(filename + ".jpeg")
def plot_image(image):      
    if False:        
        image = np.clip(image/255.0, 0.0, 1.0)
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:        
        image = np.clip(image, 0.0, 255.0)               
        image = image.astype(np.uint8)
        display(PIL.Image.fromarray(image))
def normalize_image(x):
    x_min = x.min
    x_max = x.max()
    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)    
    return x_norm
def plot_gradient(gradient):
    gradient_normalized = normalize_image(gradient) 
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()
def resize_image(image, size=None, factor=None):   
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor                
        size = size.astype(int)
    else:        
        size = size[0:2]      
    size = tuple(reversed(size))   
    img = np.clip(image, 0.0, 255.0)
    img = img.astype(np.uint8)   
    img = PIL.Image.fromarray(img)
    
    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)
    
    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized
def get_tile_size(num_pixels, tile_size=400):  
    num_tiles = int(round(num_pixels / tile_size))    
    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)    
    actual_tile_size = math.ceil(num_pixels / num_tiles)    
    return actual_tile_size
def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)
    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    x_tile_size4 = x_tile_size // 4
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    y_tile_size4 = y_tile_size // 4

  
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size
        
        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g
            
            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad


def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    # Copy the image so we don't overwrite the original image.
    img = image.copy()
    
    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)
    
    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)
        
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled
    
    return img


# In[6]:


def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
#THANK YOU HVASS!!! 

    # Do a recursive step?
    if num_repeats>0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)
            
        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)
        
        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size) 
    plot_image(img_result)
    return img_result



#starter code
session = tf.InteractiveSession(graph=model.graph)


from vidsplit import Vidsplitter
import os

#Create output dir if it doesn't exist
try:
	if not os.path.exists("output"):
		os.makedirs("output")
except OSError:
	print("Error creating directory for output")

proj_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(proj_dir, "data")
output_dir = os.path.join(proj_dir, "output")

images = Vidsplitter("video.mp4")
images.split()

print("project dir:", proj_dir)
print("data dir:", data_dir)
print("output dir:", output_dir)

for _, _, files in os.walk(data_dir):
	for i, file in enumerate(files):
		#Process only new files
		if not os.path.isfile("./output/" + file[0:-9] + "slapper.jpeg"):
			print("file", i, ":", file)
			
			image = load_image(filename="data/" + file)
			layer_tensor = model.layer_tensors[3]
			img_result = recursive_optimize(layer_tensor=layer_tensor, image=image, num_iterations=10, step_size=3.0, rescale_factor=0.7, num_repeats=4, blend=0.2)
			save_image(img_result, "output/"+str(i)+"slapper")
		else:
			print("file", i, "already processed")
