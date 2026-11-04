from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, label


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    score: float


def decode_heatmap(heatmap, config=None):
    cfg=dict(config or {})
    h=np.asarray(heatmap,dtype=float)
    if h.ndim != 2 or not np.isfinite(h).all(): raise ValueError("heatmap must be finite 2D")
    k=int(cfg.get("top_k",cfg.get("K",10))); radius=max(0,int(cfg.get("nms_radius",3))); threshold=float(cfg.get("threshold",0.0))
    if k<0: raise ValueError("top_k must be non-negative")
    if k==0: return []
    size=2*radius+1
    # Strictly exclude the exact background floor; otherwise a flat zero image
    # yields every pixel as a local maximum.
    maxima=(h==maximum_filter(h,size=size,mode="constant",cval=-np.inf)) & (h>threshold)
    components,count=label(maxima)
    peaks=[]
    for component in range(1,count+1):
        ys,xs=np.nonzero(components==component)
        best=min(range(len(xs)),key=lambda i:(-h[ys[i],xs[i]],ys[i],xs[i]))
        peaks.append(Candidate(float(xs[best]),float(ys[best]),float(h[ys[best],xs[best]])))
    peaks.sort(key=lambda c:(-c.score,c.y,c.x))
    kept=[]
    for candidate in peaks:
        if all((candidate.x-other.x)**2+(candidate.y-other.y)**2 > radius*radius for other in kept):
            kept.append(candidate)
            if len(kept)>=k: break
    return kept
