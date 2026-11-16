from dataclasses import dataclass
from pathlib import Path
import random
import yaml

@dataclass(eq=True)
class BaseConfig:
    profile: str
    angles_deg: list
    geometry: dict
    training: dict
    provenance: dict
    seed: int

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            obj = cls(**yaml.safe_load(f))
        obj.validate()
        return obj

    def save(self, path):
        from dataclasses import asdict
        self.validate()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True, sort_keys=False)

    def validate(self):
        raise NotImplementedError

def partition_ids(ids, ratios, seed):
    if not ratios or any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError('ratios must be nonnegative and sum to 1')
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    # Largest remainder allocation preserves the total count.
    exact = [len(ids)*x for x in ratios]
    sizes = [int(x) for x in exact]
    for i in sorted(range(len(sizes)), key=lambda i: exact[i]-sizes[i], reverse=True)[:len(ids)-sum(sizes)]:
        sizes[i] += 1
    result, offset = [], 0
    for n in sizes:
        result.append(ids[offset:offset+n]); offset += n
    return result
