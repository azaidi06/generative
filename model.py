from fastai.vision.all import *
import torch

def get_encoder():
    return torch.nn.Sequential(torch.nn.Conv2d(1, 28, 3, 2, 1),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(28, 64, 3, 2, 1),
                              torch.nn.ReLU(),
                              torch.nn.Conv2d(64, 128, 3, 2, 1),
                              torch.nn.ReLU(),
                              torch.nn.Flatten(),
                              torch.nn.Linear(2048, 2)
                             )


def get_decoder():
    return torch.nn.Sequential(torch.nn.Linear(2, 2048),
                                torch.nn.ReLU(),
                                # Unflatten
                                torch.nn.Unflatten(1, (128, 4, 4)),
                                # Upsample
                                torch.nn.ConvTranspose2d(128, 64, 3, 2, 1,),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(64, 28, 4, 2, 1,),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(28, 1, 4, 2, 1,)  # Use kernel of 4
                            )



class Autoencoder(torch.nn.Module): 
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()
    
    def forward(self, x):
#        pdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x


def get_learner(dls, model, device):
    if device == 'cuda':
        return Learner(dls, model, loss_func=MSELossFlat(), ).to_fp16()
    else: 
        return Learner(dls, model, loss_func=MSELossFlat())