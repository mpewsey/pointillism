# Pointillism

## About

Create pointilized images using sprites or other sub images.


## Installation

The development version of this package may be installed via pip:

```none
pip install git+https://github.com/mpewsey/pointillism#egg=pointillism
```

## Examples

The `pointilize_1` function can be used to create a pointilized
image from a base image. The following code was used to create
the image of Charizard shown below. Each pixel of the original image
has been replaced with a Pokemon whose dominant color corresponds with
the color cluster in the original image.

```python
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
```

![charizard.png](examples/charizard.png)
