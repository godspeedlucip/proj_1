from stent_repro.common import BaseConfig, partition_ids

class RunConfig(BaseConfig):
    def validate(self):
        valid = {'A-original-4view': [0,45,90,135], 'A-adapted-3view': [0,30,60]}
        if self.profile not in valid or self.angles_deg != valid[self.profile]:
            raise ValueError('profile and angles do not match')
        return self

def split_models(model_ids, seed, ratios):
    if len(ratios) != 3:
        raise ValueError('three ratios required')
    return dict(zip(('train','validation','test'), partition_ids(model_ids, ratios, seed)))

def preflight_training(cuda_available=None, peak_bytes=None, available_bytes=None):
    import torch
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError('CUDA unavailable: strict training cannot start')
    if peak_bytes is None or available_bytes is None:
        raise RuntimeError('Full-shape peak memory measurement and available memory required')
    if peak_bytes > available_bytes:
        raise RuntimeError('Insufficient GPU memory for unmodified configuration')
    return {'ready': True, 'peak_bytes': peak_bytes, 'available_bytes': available_bytes}
