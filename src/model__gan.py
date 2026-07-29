import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from typing import Optional
from transformers.file_utils import ModelOutput

@dataclass
class GANOutput(ModelOutput):
    logits: torch.FloatTensor = None
    logits_fake: Optional[torch.FloatTensor] = None
    logits_fuse: Optional[torch.FloatTensor] = None
    logits_fusef: Optional[torch.FloatTensor] = None
    logits_fuset: Optional[torch.FloatTensor] = None
    logits_fuseft: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    Dloss: Optional[torch.FloatTensor] = None
    Gloss: Optional[torch.FloatTensor] = None
    Qloss: Optional[torch.FloatTensor] = None
    asrloss: Optional[torch.FloatTensor] = None
    rep_real: Optional[torch.FloatTensor] = None
    rep_fake: Optional[torch.FloatTensor] = None
    t_: Optional[torch.FloatTensor] = None
    fc: Optional[torch.FloatTensor] = None
    logits_t: Optional[torch.FloatTensor] = None


from .vsgan import Generator, Discriminator
from .MMD import MMD
from info_nce import InfoNCE
mi = InfoNCE()

class InfoGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """
    def __init__(self, image_size, hidden_dim, z_dim, disc_dim, cont_dim, cls, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())

        self.G = Generator(image_size, z_dim, disc_dim, cont_dim)
        self.D = Discriminator(image_size, z_dim, disc_dim, cont_dim)

        self.shape = int(image_size ** 0.5)


class AudioEnc(torch.nn.Module):
    def __init__(self, model_a, config_a, num_labels, c_dim=8, ce_weight=None):
        super(AudioEnc, self).__init__()

        self.num_labels = num_labels
        self.config_a = config_a
        self.audio_wav2vec2 = model_a
        # Optional per-class weights for the InfoGAN discrete (Q) loss; moves with
        # the module via .to() but is not saved in the state_dict.
        self.register_buffer('ce_weight', ce_weight, persistent=False)
        self.gan = InfoGAN(config_a.hidden_size, 128,128, config_a.num_labels,c_dim, model_a.classifier)
        # self.others = Classifier(config_a)
        self.audio_w_sum = nn.Linear(13,1)

        self.fuse = nn.Linear(config_a.hidden_size+c_dim, config_a.num_labels)

        self.h = nn.Sequential(
                nn.Linear(config_a.hidden_size+c_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, config_a.num_labels),
        )
        self.hj = nn.Sequential(
                nn.Linear(config_a.hidden_size+c_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config_a.num_labels),
        )
        self.wmm = nn.Sequential(
                nn.Linear(config_a.hidden_size+c_dim, 1),
                nn.Sigmoid()
        )
        self.wm = nn.Linear(2,1, bias=False)
        self.ll = nn.Linear(config_a.hidden_size, config_a.num_labels)
        

        

        self.par_dic = {'D': [p for p in self.gan.D.parameters() if p.requires_grad],
                      'G': [p for p in self.gan.G.parameters() if p.requires_grad],
                    #   'Q': [p for p in self.gan.Q.parameters() if p.requires_grad],
                      'T': [p for p in self.audio_wav2vec2.wav2vec2.parameters() if p.requires_grad] +
                            [p for p in self.audio_w_sum.parameters() if p.requires_grad],
                      'C': [p for p in self.fuse.parameters() if p.requires_grad] +  [p for p in self.h.parameters() if p.requires_grad] + [p for p in self.hj.parameters() if p.requires_grad]+ [p for p in self.wm.parameters() if p.requires_grad]+ [p for p in self.ll.parameters() if p.requires_grad]+ [p for p in self.wmm.parameters() if p.requires_grad] ,
                      'M': [p for p in self.wm.parameters() if p.requires_grad] ,  
                    #   'others': [p for p in self.others.parameters() if p.requires_grad],
                      }

    def compute_noise(self, batch_size, z_dim, disc_dim, cont_dim, c=None, t=None):
        """ Compute random noise for the generator to learn to make images
        OPTIONAL: set c to explore latent dimension space.
        """

        # Noise vector (z)
        z = torch.randn(batch_size, z_dim)

        # Uniformly distributed categorical latent variables (c1)
        if c is not None:
            disc_c = c
        else:
            disc_c = torch.rand((batch_size, disc_dim))

        if t is not None:
            cont_c = t.cpu()
        else:
            cont_c = torch.randn(batch_size, cont_dim)

        return torch.cat((z, disc_c, cont_c), dim=1)
    
    def freeze_out_layer(self):
        self.audio_out_proj.requires_grad_(False)

    def forwardD(self, rep, device, c=None, alpha=0.2, t=None):


        # Sample noise z, generate output G(c, z)
        noise = self.compute_noise(rep.shape[0], self.gan.z_dim,
                                   self.gan.disc_dim, self.gan.cont_dim, c=c, t=t).to(device)
        G_output = self.gan.G(noise)


        ones = torch.ones(rep.size(0),1).to(device)
        zeros = torch.zeros(G_output.size(0),1).to(device)
        data = torch.cat((rep, G_output))
        labels = torch.cat((ones,zeros))
        perm = torch.randperm(data.size(0)).view(-1).long().to(device)
        data_, labels_ =data[perm], labels[perm]

        lam = np.random.beta(alpha, alpha)

        data = lam * data + (1. - lam) * data_
        labels = lam * labels + (1. - lam) * labels_
        
        D_score = self.gan.D(data,)[0]
        D_loss = F.binary_cross_entropy_with_logits(D_score, labels)

        # Compute vanilla (original paper) D loss
        # D_loss = -torch.mean(torch.log(DX_score + 1e-8) + torch.log(1 - DG_score + 1e-8))

        return torch.sum(D_loss)
    
    def forwardG(self, rep, device, c=None, alpha=0.2, t=None):
        # Get noise (denoted z), classify it using G, then classify the output of G using D.
        noise = self.compute_noise(rep.shape[0], self.gan.z_dim,
                                   self.gan.disc_dim, self.gan.cont_dim, c=c, t=t).to(device) # c=[c1, c2], z
        G_output = self.gan.G(noise) # G(c, z)

        ones = torch.ones(G_output.size(0),1).to(device)
        zeros = torch.zeros(rep.size(0),1).to(device)
        data = torch.cat((G_output, rep))
        labels = torch.cat((ones,zeros))
        perm = torch.randperm(data.size(0)).view(-1).long().to(device)
        data_, labels_ =data[perm], labels[perm]

        lam = np.random.beta(alpha, alpha)

        data = lam * data + (1. - lam) * data_
        labels = lam * labels + (1. - lam) * labels_
        
        D_score = self.gan.D(data,)[0]
        G_loss = F.binary_cross_entropy_with_logits(D_score, labels)

        return G_loss
    
    def forwardQ(self, rep, device, LAMBDA=1, c=None, t=None):
        # Generate labels for the generator batch rep (all 0, since they are fake)
        G_labels = torch.zeros(rep.shape[0], 1)

        # Sample noise z, generate output G(c, z)
        noise = self.compute_noise(rep.shape[0], self.gan.z_dim,
                                   self.gan.disc_dim, self.gan.cont_dim,c=c, t=t).to(device)

        # Transform noise using G
        G_output = self.gan.G(noise)

        # Approximate true posterior for categorical, Gaussian latent variables
        Q_continuous, Q_discrete = self.gan.D(G_output)[2:4]

        # Compute mutual information loss
        # Discrete component
        discrete_target = noise[:, self.gan.z_dim:self.gan.z_dim+self.gan.disc_dim]
        disc_loss = F.cross_entropy(Q_discrete, torch.max(discrete_target, 1)[1], weight=self.ce_weight)

        # Continuous component
        continuous_target = noise[:, self.gan.z_dim+self.gan.disc_dim:]
        cont_loss = mi(Q_continuous, continuous_target)

        # Sum it up
        MI_loss = LAMBDA * (disc_loss + cont_loss)

        return MI_loss, disc_loss, cont_loss

    def forward(self, data, device, t=None):

        c= data['labels'].view(-1, self.num_labels)
        # c= torch.argmax(data['labels'], dim=-1).squeeze().cpu().detach().numpy().tolist()
        # c=None
       
        out_audio = self.audio_wav2vec2(
            input_values=data['input_values'].to(device), 
            # labels=data['labels_asr'].to(device), 
            # output_hidden_states=True 
            )

        feat_audio = out_audio[0]
        Dloss = self.forwardD(feat_audio.detach(), device, c=c, t=t)
        Gloss = self.forwardG(feat_audio.detach(), device, c=c, t=t)
        Qloss = self.forwardQ(feat_audio.detach(), device, c=c, t=t)

        t_, logit_audio = self.gan.D(feat_audio)[2:4]

        if t is not None:
            logits_fuse = self.hj(torch.concat((feat_audio.view(feat_audio.shape[0],-1), t),dim=1))
        else:
            logits_fuse = None
        logits_fuset = self.hj(torch.concat((feat_audio.view(feat_audio.shape[0],-1), t_),dim=1))
        
        
        noise = self.compute_noise(feat_audio.shape[0], self.gan.z_dim,
                                self.gan.disc_dim, self.gan.cont_dim,c=c, t=t).to(device)    
        c_=noise[:, self.gan.z_dim:self.gan.z_dim+self.gan.disc_dim]
        c_  = torch.randn_like(c_)# *3-1
        fake = self.gan.G(noise)
        l = MMD(feat_audio.view(feat_audio.shape[0],-1), fake.view(fake.shape[0], -1), device)

        logit_fake = self.gan.D(fake)[3]
        if t is not None:
            logits_fusef = self.hj(torch.concat((fake.view(fake.shape[0],-1), t),dim=1))
        else:
            logits_fusef = None
        
        logits_fuseft = self.hj(torch.concat((fake.view(fake.shape[0],-1), t_),dim=1))
        
        
        return GANOutput(
                logits=logit_audio,
                logits_fake=logit_fake,
                logits_fuse=logits_fuse,
                logits_fusef=logits_fusef,
                logits_fuset=logits_fuset,
                logits_fuseft=logits_fuseft,
                loss=l,
                Dloss=Dloss,
                Gloss=Gloss,
                Qloss=Qloss,
                rep_real=feat_audio.view(feat_audio.shape[0],-1),
                rep_fake=fake.view(fake.shape[0],-1),
                t_=t_,
                fc=c_,
                # logits_t=logit_t,
            )
    
