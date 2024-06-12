from gan import GAN
from generator import Generator
from discriminator import Discriminator
from gan import GAN

import tensorflow as tf

disc = Discriminator([1080, 720, 3], 2, None, None)
gen = Generator([1080, 720, 3], 2, None, None)

gen.create_model()
disc.create_model()

disc.print_model_summary()
gen.print_model_summary()

network = GAN([720, 1080, 3], None, None)
network.create_model(generator=gen.model, discriminator=disc.model)
network.print_model_summary()