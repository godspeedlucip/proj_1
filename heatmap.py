import numpy as np


def make_heatmap(marker_xy, image_size, sigma=2.0):
    h,w=map(int,image_size)
    if h<=0 or w<=0 or sigma<=0: raise ValueError("image_size and sigma must be positive")
    points=np.asarray(marker_xy,dtype=float)
    if points.shape != (2,2) or not np.isfinite(points).all(): raise ValueError("marker_xy must be finite 2x2")
    yy,xx=np.mgrid[:h,:w]
    maps=[np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma)) for x,y in points]
    return np.maximum(maps[0],maps[1]).astype(np.float32)
