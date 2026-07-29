import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, image_size, z_dim, disc_dim, cont_dim):
        super(Generator, self).__init__()
        input_dim = z_dim + disc_dim + cont_dim

        self.l1 = nn.Sequential(nn.Linear(input_dim, image_size))


    def forward(self, gen_input):
        # gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size, z_dim, disc_dim, cont_dim):
        super(Discriminator, self).__init__()

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(image_size, 1))
        self.aux_layer = nn.Sequential(nn.Linear(image_size, disc_dim))
        self.latent_layer = nn.Sequential(nn.Linear(image_size, cont_dim))
        self.cls_layer = nn.Sequential(nn.Linear(image_size, disc_dim))
        self.adv_layerc = nn.Sequential(nn.Linear(disc_dim, 1))
        self.adv_layerd = nn.Sequential(nn.Linear(cont_dim, 1))

    def forward(self, img, c=None, t=None):
        validity = self.adv_layer(img)
        if c is not None:
            validity+=self.adv_layerc(c)
        if t is not None:
            validity+=self.adv_layerd(t)
        label = self.aux_layer(img)
        latent_code = self.latent_layer(img)
        cls = self.cls_layer(img)

        return validity, label, latent_code, cls