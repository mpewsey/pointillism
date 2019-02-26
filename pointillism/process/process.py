import cv2
import imageio
import numpy as np
import os, glob, random
from collections import Counter
from sklearn.cluster import KMeans

__all__ = [
    'crop_image',
    'dominant_color',
    'create_grid',
    'make_palette',
    'pointilize_1',
    'pointilize_2',
]


def crop_image(img):
    """
    Returns an image array cropped to exclude transparent margins.

    Parameters
    ----------
    img : array
        A 2D image array of RGB or RGBA values.
    """
    img = np.asarray(img)
    xl, xu = float('inf'), -float('inf')
    yl, yu = float('inf'), -float('inf')

    for j, y in enumerate(img):
        for i, x in enumerate(y):
            if len(x) < 4 or x[3] > 0:
                xl, xu = min(i, xl), max(i, xu)
                yl, yu = min(j, yl), max(j, yu)

    dx = xu - xl + 1
    dy = yu - yl + 1

    return img[yl:yl+dy,xl:xl+dx]


def dominant_color(img, n=3):
    """
    Returns the dominant color

    Parameters
    ----------
    img : array
        An 2D image array of RGB or RGBA values.
    n : int
        The number of color clusters to apply.
    """
    img = np.asarray(img)
    img = img.reshape(-1, img.shape[-1])

    if img.shape[-1] > 3:
        img = img[:,:3][img[:,3] > 0]
    else:
        img = img[:,:3]

    cluster = KMeans(n_clusters=n)
    labels = cluster.fit_predict(img)

    counts = Counter(labels)
    (counts, _), = counts.most_common(1)

    color = cluster.cluster_centers_[counts]
    color = tuple(map(int, color))

    return color


def create_grid(img, n=7):
    """
    Creates a 2D array of 'paint by number' integers for an image. The number
    of values corresponds to the number of clusters specified. If an index
    has a transparency value of 0, a value of -1 will be returned for that
    index.

    Parameters
    ----------
    img : array
        A 2D image array of RGB or RGBA values.
    n : int
        The number of color clusters to identify.
    """
    img = np.asarray(img)
    shape = img.shape[:2]
    cluster = KMeans(n_clusters=n)

    img = img.reshape(-1, img.shape[-1])
    labels = cluster.fit_predict(img[:,:3])

    if img.shape[-1] > 3:
        labels[img[:,3] == 0] = -1

    labels = labels.reshape(shape)

    return labels, cluster


def make_palette(sprites, n=7):
    """
    Creates a palette of different sprite clusters.

    Parameters
    ----------
    sprites : list
        A list of sprite file paths for which the palette will be created.
    n : int
        The number of color clusters to apply for palette creation.
    """
    cluster = KMeans(n_clusters=n)

    colors = np.array([dominant_color(imageio.imread(s)) for s in sprites])
    colors = cluster.fit_predict(colors)

    palette = [[] for _ in range(n)]

    for i, sprite in zip(colors, sprites):
        palette[i].append(sprite)

    return palette


def pointilize_1(img, sprites, dim=100, margin=0, n=7, background=(0, 0, 0, 255)):
    """
    Uses sprites or images to create a pointilized image from a base image.
    Returns a 2D image array of RGBA values representing the pointilized image.

    Parameters
    ----------
    img : array
        A 2D image array of RGB or RGBA values.
    sprites : list
        A list of sprite or image file paths.
    dim : int
        The width and height in pixels of the image applied at each grid index.
    margin : int
        The inner margin to apply to the image in each grid index.
    background : list
        The RGBA value to apply to the image background.
    """
    img, cluster = create_grid(img, n)

    # Create a sprite palette for the main image clusters
    colors = np.array([dominant_color(imageio.imread(s)) for s in sprites])
    colors = cluster.predict(colors)

    palette = [[] for _ in range(n)]

    for i, sprite in zip(colors, sprites):
        palette[i].append(sprite)

    return pointilize_2(img, palette, dim, margin, background)


def pointilize_2(grid, palette, dim=100, margin=0, background=(0, 0, 0, 255)):
    """
    Using a grid of 'paint by number' integers, creates an image of sprites
    or sub images based on the provided palette.

    Parameters
    ----------
    grid : array
        A 2D array of integers used to place sprite palette contents.
        If less than zero, the index will be ignored.
    palette : list
        A list of lists of sprite paths. Each index should correspond to
        a number in the provided grid.
    dim : int
        The width and height in pixels of the image applied at each grid index.
    margin : int
        The inner margin to apply to the image in each grid index.
    background : list
        The RGBA value to apply to the image background.
    """
    grid = np.asarray(grid, dtype='int')
    img = np.zeros((dim*grid.shape[0], dim*grid.shape[1], 4), dtype='int')
    off = int(dim / 2)

    # Loop through the rows and columns of the array and insert a resized
    # and centered sprite image.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] >= 0:
                p = palette[grid[i, j]]
                p = p[random.randint(0, len(p)-1)]

                pix = imageio.imread(p)

                w = (dim - 2 * margin) / pix.shape[1]
                h = (dim - 2 * margin) / pix.shape[0]

                scale = min(w, h)

                w = int(pix.shape[1] * scale)
                h = int(pix.shape[0] * scale)

                pix = cv2.resize(pix, (w, h))

                w = int(pix.shape[1] / 2)
                h = int(pix.shape[0] / 2)

                w = off - w + j * dim
                h = off - h + i * dim

                img[h:h+pix.shape[0],w:w+pix.shape[1]] = pix

    # Assign the background color to non transparent pixels and blue the image
    shape = img.shape
    img = img.reshape(-1, 4)
    img[img[:,3] < 255] = background
    img = img.reshape(*shape)
    img = cv2.blur(img, (2, 2))

    return np.asarray(img, dtype='uint8')
