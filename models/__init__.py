# __init__.py

from .wgan_gp import WGANGP
from .dcgan import DCGAN

from .generator import Generator
from .discriminator import Discriminator
from .critic import Critic

__all__ = [
    'WGANGP',
    'DCGAN',
    'Generator',
    'Discriminator',
    'Critic',
]