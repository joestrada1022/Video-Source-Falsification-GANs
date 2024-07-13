# __init__.py

from .cgan import CGAN
from .wgan import WGAN
from .wcgan import WCGAN

from .generator import Generator
from .discriminator import Discriminator

__all__ = [
    'CGAN',
    'WGAN',
    'WCGAN',
    'Generator',
    'Discriminator'
]