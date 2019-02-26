# charizard.py
import random
import imageio
from pointillism import pointilize_1, sprite_paths

# Set the random seed for repeatable results
random.seed(12345)

# Get a list of sprite file paths
sprites = sprite_paths('pokemon')

# Load the base image
img = imageio.imread(sprites[5])

# Pointilize the base image with the sprites
img = pointilize_1(img, sprites, n=5)

# Write the result to file
imageio.imwrite('charizard.png', img)
