from gan_training.models import (dcgan_deep, dcgan_shallow, resnet2, biggan) #

generator_dict = {
    'resnet2': resnet2.Generator,
    'dcgan_deep': dcgan_deep.Generator,
    'dcgan_shallow': dcgan_shallow.Generator,
    'biggan': biggan.Generator
}

discriminator_dict = {
    'resnet2': resnet2.Discriminator,
    'dcgan_deep': dcgan_deep.Discriminator,
    'dcgan_shallow': dcgan_shallow.Discriminator,
    'biggan': biggan.Discriminator
}
