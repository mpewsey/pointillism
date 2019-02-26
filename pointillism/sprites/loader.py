import os
import glob

__all__ = ['sprite_paths']


SPRITES_FOLDER = os.path.abspath(os.path.dirname(__file__))


def sprite_paths(name):
    """
    Returns a list of all files located at the specified namespace within
    the sprites folder.

    Parameters
    ----------
    name : str
        The name of the sprite namespace.
    """
    path = os.path.join(SPRITES_FOLDER, name, '*.png')
    return glob.glob(path)
