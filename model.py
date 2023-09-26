from fastai.vision.all import *
import torch


def get_learner(dls, model, device, loss_func=MSELossFlat()):
    if device == 'cuda': #lets just default to using FP16 if we have GPU
        return Learner(dls, model, loss_func=loss_func, ).to_fp16()
    else: 
        return Learner(dls, model, loss_func=loss_func,)
    
# Module class to get an encoder with Relu/BatchNorm
# Pass a list of numbers indicating how many channels in each layer
class Encoder(torch.nn.Module):
    def __init__(self, conv_channels=[1,28,64,128]):
        super(Encoder, self).__init__()
        self.int_conv_layers = conv_channels
        self.conv_body = self.get_conv_layers()
        
    def get_conv_layers(self, flatten=True):
        conv_layers = torch.nn.ModuleList()
        for x in range(len(self.int_conv_layers)-1):
            in_ch = self.int_conv_layers[x]
            out_ch = self.int_conv_layers[x+1]
            conv_layers.append(self.get_conv(in_ch, out_ch))
        if flatten: conv_layers.append(torch.nn.Flatten())
        return torch.nn.Sequential(*conv_layers)
                                                    
    def get_conv(self, in_ch, out_ch, activation=True, 
                 batch_norm=True):
        conv_layer = torch.nn.ModuleList([torch.nn.Conv2d(in_ch, out_ch, 
                                                          kernel_size=3, 
                                                          stride=2, 
                                                          padding=1),])
        if activation: conv_layer.append(torch.nn.ReLU())
        if batch_norm: conv_layer.append(torch.nn.BatchNorm2d(out_ch))
        return torch.nn.Sequential(*conv_layer)
    
    def forward(self, x):
        return self.conv_body(x)
    
    
class Decoder(torch.nn.Module):
    def __init__(self, conv_channels=[128,64,28,1], in_dims=2):
        super(Decoder, self).__init__()
        self.int_conv_layers = conv_channels
        self.in_dims = in_dims
        self.linear = torch.nn.Linear(in_dims, 2048)
        self.act = torch.nn.ReLU()
        self.unflatten = torch.nn.Unflatten(1, (128, 4, 4))
        self.conv_body = self.get_conv_layers()

        
    def get_conv_layers(self, flatten=True):
        conv_layers = torch.nn.ModuleList()
        for x in range(len(self.int_conv_layers)-1):
            in_ch = self.int_conv_layers[x]
            out_ch = self.int_conv_layers[x+1]
            conv_layers.append(self.get_conv(in_ch, out_ch))
        #if flatten: conv_layers.append(torch.nn.Flatten())
        return torch.nn.Sequential(*conv_layers)
                                                    
    
    def get_conv(self, in_ch, out_ch, activation=True, 
                 batch_norm=True, ks=4):
        if in_ch==128: ks=3
        conv_layer = torch.nn.ModuleList([torch.nn.ConvTranspose2d(in_ch, 
                                                                   out_ch, 
                                                                   kernel_size=ks, 
                                                                   stride=2, 
                                                                   padding=1),])
        if activation: conv_layer.append(torch.nn.ReLU())
        if batch_norm: conv_layer.append(torch.nn.BatchNorm2d(out_ch))
        return torch.nn.Sequential(*conv_layer)
    
    def forward(self, x):
        x = self.unflatten(self.act(self.linear(x)))
        return self.conv_body(x)
    
    
class Autoencoder(torch.nn.Module): 
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.lin = torch.nn.Linear(2048, 2)
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        x = self.lin(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    

class Vae(torch.nn.Module): 
    def __init__(self, latent_dims=2):
        super(Vae, self).__init__()
        self.latent_dims = latent_dims 
        self.encoder = Encoder()
        self.mean_lin = torch.nn.Linear(2048, self.latent_dims)
        self.var_lin = torch.nn.Linear(2048, self.latent_dims
                                      
                                      
                                      
                                      )
        self.sampler = VaeSampler()
        self.decoder = Decoder(in_dims=self.latent_dims)
    
    def forward(self, x):
        z_mean, z_var = self.encode(x)
        z = self.sampler(z_mean, z_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_var
    
    
    def encode(self, x):
        x = self.encoder(x)
        #pdb.set_trace()
        z_mean = self.mean_lin(x)
        z_var = self.var_lin(x)
        return z_mean, z_var
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def generate(self, x):
        return self.decoder(self.sampler(*self.encode(x)))
    
    
class VaeSampler(torch.nn.Module):
    def __init__(self):
        super(VaeSampler, self).__init__()
    
    def forward(self, z_mean, z_var):
        device = 'cuda' if z_mean.device.type == 'cuda' else 'cpu'
        epsilon = torch.normal(mean=torch.tensor(0.0),
                               std=torch.tensor(1.0),
                               size=(z_mean.shape[0],
                                     z_mean.shape[1])).to(device)
        z = z_mean + torch.exp(0.5 * z_var.float()) * epsilon
        return z
    
    
class FeatureLoss(Module):
    def __init__(self, cross_entropy=True):
        if cross_entropy: self.recon_loss = torch.nn.BCEWithLogitsLoss()
        else: self.recon_loss = torch.nn.MSELoss()
        self.kl_loss = KlLoss()

    def forward(self, preds, ys):
        reconstruction, z_mean, z_var = preds
        reconstruction_loss = self.recon_loss(reconstruction, ys)
        kl_loss = self.kl_loss(z_mean, z_var)
        total_loss = reconstruction_loss + kl_loss / 64
        return total_loss

    
class KlLoss(Module):
    def __init__(self):
        pass
    def forward(self, z_mean, z_var):
        return torch.sum(-0.5 * (1 + z_var - z_mean**2 - torch.exp(z_var)))