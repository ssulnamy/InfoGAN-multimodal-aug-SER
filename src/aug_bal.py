import torch
import numpy as np
import torch.nn.functional as F

def aug_bal(model, labels, device):
    l = torch.sum(labels, dim=0).detach().cpu().numpy()
    mci = int(np.argmax(l, axis=-1))
    l = np.max(l) - l
    i = 0
    ls = []
    while True:
        # if i == mci:
        #     i += 1
        #     continue
        if i > 3:
            break
        if l[i] <=0 :
            i +=1
            continue
        nl = np.zeros_like(l)
        nl[i] = 1.
        l = l - nl
        ls.append(torch.tensor(nl))
        # if np.sum(l) <= 0:
        #     break
    if len(ls) > 0:
        new_l = torch.stack(ls, dim=0)
    else:
        new_l = labels.cpu()
    total_loss = 0.0

    for n in range(4):

        noise = model.compute_noise(new_l.shape[0], model.gan.z_dim,
                                model.gan.disc_dim, model.gan.cont_dim,c=new_l).to(device)    
        fake = model.gan.G(noise)
        t_ = model.gan.D(fake)[2]

        logits = model.hj(torch.concat((fake.detach(), t_.detach()),dim=1))
        loss = F.cross_entropy(logits, new_l.to(device))
        total_loss += loss
    for n in range(20):

        random_index = torch.randint(0, 4, (new_l.shape[0],))
        aug_l = F.one_hot(random_index, num_classes=4).float()

        # aug_l = torch.rand_like(new_l)

        noise = model.compute_noise(new_l.shape[0], model.gan.z_dim,
                                model.gan.disc_dim, model.gan.cont_dim,c=aug_l).to(device)    
        fake = model.gan.G(noise)
        t_ = model.gan.D(fake)[2]

        logits = model.hj(torch.concat((fake.detach(), t_.detach()),dim=1))
        loss = F.cross_entropy(logits, aug_l.to(device))
        total_loss += loss
    return total_loss