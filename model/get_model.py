import torch.nn as nn
from model.model_zoo import GoalGAN, GoalGANDiscriminator

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_model(cfg):
    """
    Loads Generator and Discriminator
    """

    generator = GoalGAN(**cfg)
    generator.apply(init_weights)
    discriminator = GoalGANDiscriminator(**cfg )
    discriminator.apply(init_weights)
    return generator, discriminator
