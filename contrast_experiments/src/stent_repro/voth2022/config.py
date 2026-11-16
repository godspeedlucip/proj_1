from stent_repro.common import BaseConfig
import random

class RunConfig(BaseConfig):
    def validate(self):
        valid = {'B-original-3view': [0,60,120], 'B-adapted-3view': [0,30,60], 'B-one-channel-ablation': [0,60,120]}
        if self.profile not in valid or self.angles_deg != valid[self.profile]:
            raise ValueError('profile and angles do not match')
        return self

def split_scenes(scene_ids, model_ids, ratios, seed):
    if len(scene_ids) != len(model_ids) or len(set(scene_ids)) != len(scene_ids):
        raise ValueError('unique scene IDs and model IDs per scene required')
    if len(ratios) != 2 or any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('two nonnegative ratios summing to 1 required')
    models = [[m] if isinstance(m, str) else list(m) for m in model_ids]
    if any(not group or any(not isinstance(m,str) or not m for m in group) for group in models):
        raise ValueError('each scene requires nonempty model IDs')
    parents = list(range(len(scene_ids)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owner = {}
    for i, group in enumerate(models):
        for model in group:
            if model in owner: parents[find(i)] = find(owner[model])
            else: owner[model] = i
    components = {}
    for i in range(len(scene_ids)): components.setdefault(find(i), []).append(i)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    result = {'train': [], 'validation': []}
    names = ('train','validation')
    for group in groups:
        # Choose the assignment closest to the requested scene counts.
        def cost(k):
            counts = [len(result[n]) + (len(group) if j == k else 0) for j,n in enumerate(names)]
            return sum((counts[j]-len(scene_ids)*ratios[j])**2 for j in range(2))
        chosen = min(range(2), key=cost)
        for i in group:
            row = {'scene_id':scene_ids[i], 'model_ids':models[i]}
            if len(models[i]) == 1: row['model_id'] = models[i][0]
            result[names[chosen]].append(row)
    return result

def preflight_memory(cuda_available=None, peak_bytes=None, available_bytes=None):
    import torch
    if cuda_available is None: cuda_available = torch.cuda.is_available()
    if not cuda_available: return {'ready': False, 'reason': 'CUDA unavailable'}
    if peak_bytes is None or available_bytes is None: return {'ready': False, 'reason': 'Full-shape peak and available memory must be measured'}
    return {'ready': peak_bytes <= available_bytes, 'reason': 'measured', 'peak_bytes': peak_bytes, 'available_bytes': available_bytes}
