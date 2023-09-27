from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import *


def plot_img(ax, path):
    ax.imshow(np.array(Image.open(path)))
    ax.axis('off')
    
    
def plot_dataset(df, num_samples=6):
    plt.figure(figsize=(4,4))
    if num_samples > 6: num_samples=6
    rand_idxs = np.random.randint(1, len(df), num_samples)
    paths = df.path.iloc[rand_idxs].values
    for idx, path in enumerate(paths):
        ax = plt.subplot(1, num_samples, idx+1)
        plot_img(ax, path)

        
def get_valid_results(model, dl, num_samples=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch = next(iter(dl))
    if get_model_type(model): #if it's a VAE, diff fwd pass
        output, _, _ = model(batch[0].to(device))
        output = output.detach().cpu()
    else:
        output = model(batch[0].to(device)).detach().cpu()
    return batch[0][:num_samples].cpu(), output[:num_samples]


def get_subplot(img, row=1, cols=2, item=1):
    ax = plt.subplot(row, cols, item)
    ax.imshow(np.array(img.squeeze()))
    ax.axis('off')
    return ax


def plt_subs(og_imgs, gen_imgs, size=3):
    rows = int(len(og_imgs) / 2)
    cols = 4
    fig = plt.figure(figsize=(size*2, size*rows))

    for idx, imgs in enumerate(zip(og_imgs, gen_imgs)):
        ax1 = get_subplot(imgs[0], row=rows, cols=cols, item=2*idx+1)
        ax2 = get_subplot(imgs[1], row=rows, cols=cols, item=2*idx+2)
        if idx == 0 or idx == 1:
            ax1.set_title("Original Images", fontdict={'fontsize':7})
            ax2.set_title("Generated Images", fontdict={'fontsize':7})
    plt.tight_layout(w_pad=0.5, h_pad=0.5)
    fig.subplots_adjust(top=0.5)
    

def get_random_latent_vals(min=-30, max=30, rows=6, cols=2):
    return min + (max - min) * np.random.rand(rows, cols)


#def get_regen(model, x=2, y=20):
#    return model.decode(torch.tensor([x,y]).float().unsqueeze(0))

def get_regen(model, x):
    return model.decode(torch.tensor(x).float().unsqueeze(0))
    

def get_embeds(model, dl):
    vae = get_model_type(model)
    if vae:
        embeds = [model.sampler(*model.encode(batch[0].to('cpu'))) for batch in dl]
    else:
        embeds = [model.encode(batch[0]) for batch in dl]
    embeds = torch.vstack(embeds[:-1]).detach().numpy()
    lbls = [batch[1] for batch in dl]
    lbls = np.hstack([np.array(x).astype('int') for x in lbls[:-1]])
    return embeds, lbls


def get_model_type(model):
    try: 
        model.__getattr__('sampler')
        vae_model = True
    except AttributeError:
        vae_model=False
    return vae_model


def get_encoded(model, train=False, shuffle=True):
    dl = get_dl(train=train, bs=512, shuffle=True, num_workers=8)
    embeds, lbls = get_embeds(model, dl)
    return embeds, lbls


def get_decoded(model, latents=None, min=-30, max=30, rows=3, cols=2):
    if latents is None: 
        latents = get_random_latent_vals(min=min, max=max, rows=rows, cols=cols)
    #imgs = torch.stack([get_regen(model, *pt).squeeze(0) for pt in latents])
    imgs = torch.stack([get_regen(model, pt) for pt in latents])
    return imgs, latents


def plot_2d_latent_regen(model, latents=None, min=-30, max=30):
    model.to('cpu');
    cols = model.decoder.linear.in_features
    # lets get our necessary items
    embeds, lbls = get_encoded(model)
    regens, latents = get_decoded(model, latents=latents, min=min, max=max, cols=cols)

    fig = plt.figure(figsize=(12, 6))
    
    # Define the GridSpec layout
    gs = GridSpec(1, 2, width_ratios=[1.25, 1])

    # Create the scatter plot on the left
    ax0 = fig.add_subplot(gs[0])
    sc = ax0.scatter(embeds[:, 0], embeds[:, 1],c=lbls, s=3, cmap='rainbow')
    ax0.set_title("Latents/Embeddings")
    cbar = plt.colorbar(sc, ax=ax0)
    
    #background black marker (bigger) and then white one -- for better contrast
    ax0.scatter(latents[:,0], latents[:,1], marker='x', s=200, linewidths=7, color='black')
    ax0.scatter(latents[:,0], latents[:,1], marker='x', s=200, linewidths=5, color='white')
    
    # Create three smaller subplots on the right
    for i in range(3):
        ax = fig.add_subplot(3, 2, 2*i + 2)
        ax.imshow(regens[i].detach().numpy().squeeze())
        ax.set_title([f'{x:.2f}' for x in latents[i][:2]], fontdict={'fontsize':12})
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_latent_regen(model, latents=None, min=-3, max=3):
    model.to('cpu');
    cols = model.decoder.linear.in_features
    # lets get our necessary items
    embeds, lbls = get_encoded(model)
    regens, latents = get_decoded(model, latents=latents, min=min, max=max, cols=3)

    fig = plt.figure(figsize=(12, 6))

    # Define the GridSpec layout
    gs = GridSpec(1, 2, width_ratios=[1.25, 1])

        # Create the scatter plot on the left
    ax0 = fig.add_subplot(gs[0], projection='3d')
    sc = ax0.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2] ,c=lbls, s=0.2, cmap='rainbow', zorder=1)
    ax0.set_title("Latents/Embeddings")
    cbar = plt.colorbar(sc, ax=ax0)

    #background black marker (bigger) and then white one -- for better contrast
    ax0.scatter(latents[:,0], latents[:,1], latents[:,2], marker='x', s=200, linewidths=8, color='red', zorder=2)
    ax0.scatter(latents[:,0], latents[:,1], latents[:,2],  marker='x', s=200, linewidths=5.5, color='white', zorder=2)

    # Create three smaller subplots on the right
    for i in range(3):
        ax = fig.add_subplot(3, 2, 2*i + 2)
        ax.imshow(regens[i].detach().numpy().squeeze())
        ax.set_title([f'{x:.2f}' for x in latents[i][:3]], fontdict={'fontsize':12})
        ax.axis('off')