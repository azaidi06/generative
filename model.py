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
    def __init__(self, conv_channels=[128,64,28,1]):
        super(Decoder, self).__init__()
        self.int_conv_layers = conv_channels
        self.linear = torch.nn.Linear(2, 2048)
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
        self.encoder = Encoder()#get_encoder()
        self.lin = torch.nn.Linear(2048, 2)
        self.decoder = Decoder()#get_decoder()
    
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