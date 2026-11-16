from stent_repro.networks import UNet

class A_DTE(UNet):
    def __init__(self):
        super().__init__(2, 1, (64,128,256,512), dropout=0.2, final='relu')

class A_DTR(UNet):
    def __init__(self, C=5):
        if C not in (1,3,5): raise ValueError('C must be 1, 3, or 5')
        super().__init__(2, C, (64,128,256,512), dropout=0.2, final='sigmoid')

def create_dte(): return A_DTE()
