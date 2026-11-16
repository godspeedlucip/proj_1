from stent_repro.networks import UNet

class B_DTE(UNet):
    def __init__(self):
        super().__init__(2, 1, (64,128,256,512,1024,2048,4096), final='relu')

class B_DTR(UNet):
    def __init__(self, input_channels=3):
        if input_channels not in (1,3): raise ValueError('input_channels must be 1 or 3')
        super().__init__(3, input_channels, (16,32,64,128,256), final='sigmoid')
