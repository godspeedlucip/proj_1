"""Small CPU geometry reference. Coordinates are millimetres in x,y,z; arrays are z,y,x."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates

@dataclass(frozen=True)
class Grid:
    shape: tuple  # z,y,x
    spacing_mm: tuple  # x,y,z
    origin_mm: tuple  # x,y,z coordinate of index (0,0,0)

@dataclass(frozen=True)
class ConeBeamGeometry:
    source: np.ndarray
    detector_center: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    pixel_size_mm: float
    detector_shape: tuple  # height,width

def circular_view(angle_deg, sdd_mm, sid_mm, detector_shape, pixel_size_mm):
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    return ConeBeamGeometry(rot @ np.array([0.,-sid_mm,0.]), rot @ np.array([0.,sdd_mm-sid_mm,0.]), rot @ np.array([1.,0.,0.]), np.array([0.,0.,1.]), pixel_size_mm, detector_shape)

def project_points(points, view):
    points = np.asarray(points, dtype=float)
    normal = np.cross(view.u_axis, view.v_axis)
    direction = points - view.source
    denominator = direction @ normal
    numerator = (view.detector_center - view.source) @ normal
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = numerator / denominator
        hit = view.source + scale[...,None]*direction - view.detector_center
    h,w = view.detector_shape
    uv = np.stack((hit @ view.u_axis / view.pixel_size_mm + (w-1)/2, hit @ view.v_axis / view.pixel_size_mm + (h-1)/2),axis=-1)
    return np.where((scale > 0)[...,None],uv,np.nan)

def backproject_each(projections, views, grid, chunk_depth=16):
    projections = np.asarray(projections)
    if len(views) != len(projections) or any(tuple(p.shape) != tuple(v.detector_shape) for p,v in zip(projections,views)):
        raise ValueError('projection count and shape must match views in order')
    if chunk_depth < 1: raise ValueError('chunk_depth must be positive')
    nz,ny,nx = grid.shape
    sx,sy,sz = grid.spacing_mm; ox,oy,oz = grid.origin_mm
    out = np.zeros((len(views),nz,ny,nx),dtype=np.float32)
    for start in range(0,nz,chunk_depth):
        z,y,x = np.meshgrid(oz+np.arange(start,min(start+chunk_depth,nz))*sz, oy+np.arange(ny)*sy, ox+np.arange(nx)*sx,indexing='ij')
        points = np.stack((x,y,z),axis=-1)
        for i, view in enumerate(views):
            uv = project_points(points,view)
            coords = np.stack((np.nan_to_num(uv[...,1],nan=-1e6), np.nan_to_num(uv[...,0],nan=-1e6)))
            out[i,start:start+len(z)] = map_coordinates(projections[i].astype(float),coords,order=1,mode='constant',cval=0)
    return out

def forward_project(volume, grid, views, step_mm=None, ray_chunk=512):
    """CPU midpoint ray integration of a scalar attenuation field (1/mm -> line integral)."""
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != tuple(grid.shape) or not np.isfinite(volume).all(): raise ValueError('volume shape or values invalid')
    spacing = np.asarray(grid.spacing_mm, dtype=float); origin = np.asarray(grid.origin_mm, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0): raise ValueError('invalid grid spacing')
    if step_mm is None: step_mm = float(spacing.min()/2)
    if not np.isfinite(step_mm) or step_mm <= 0 or ray_chunk < 1: raise ValueError('invalid integration step or chunk')
    lower = origin - spacing/2
    upper = origin + (np.array(grid.shape[::-1])-.5)*spacing
    results = []
    for view in views:
        h,w=view.detector_shape
        vv,uu=np.meshgrid((np.arange(h)-(h-1)/2)*view.pixel_size_mm,(np.arange(w)-(w-1)/2)*view.pixel_size_mm,indexing='ij')
        targets=view.detector_center+uu[...,None]*view.u_axis+vv[...,None]*view.v_axis
        dirs=targets.reshape(-1,3)-view.source
        max_dist=np.linalg.norm(dirs,axis=1); dirs=dirs/max_dist[:,None]
        out=np.zeros(len(dirs),np.float32)
        for begin in range(0,len(dirs),ray_chunk):
            d=dirs[begin:begin+ray_chunk]; dist=max_dist[begin:begin+ray_chunk]
            with np.errstate(divide='ignore',invalid='ignore'):
                t0=(lower-view.source)/d; t1=(upper-view.source)/d
            parallel=np.abs(d)<1e-15
            outside=parallel & ((view.source < lower)|(view.source > upper))
            tmin=np.where(parallel,-np.inf,np.minimum(t0,t1)); tmax=np.where(parallel,np.inf,np.maximum(t0,t1))
            start=np.maximum(np.max(tmin,axis=1),0); end=np.minimum(np.min(tmax,axis=1),dist)
            length=np.maximum(end-start,0); length[np.any(outside,axis=1)]=0
            counts=np.maximum(np.ceil(length/step_mm).astype(int),1)
            nmax=int(counts.max()); jj=np.arange(nmax)[None,:]
            t=start[:,None]+(jj+.5)*length[:,None]/counts[:,None]
            xyz=view.source+d[:,None,:]*t[...,None]
            ijk=(xyz-origin)/spacing
            values=map_coordinates(volume,[ijk[...,2],ijk[...,1],ijk[...,0]],order=1,mode='grid-constant',cval=0.0)
            values[jj>=counts[:,None]]=0
            out[begin:begin+len(d)]=values.sum(axis=1)*length/counts
        results.append(out.reshape(h,w))
    return np.stack(results)


def _angular_weights(angles_deg, mode):
    a=np.mod(np.deg2rad(np.asarray(angles_deg,dtype=float)),2*np.pi)
    if a.ndim != 1 or len(a)<2 or not np.isfinite(a).all(): raise ValueError('at least two finite angles required')
    order=np.argsort(a); ordered=a[order]; gaps=np.diff(np.r_[ordered,ordered[0]+2*np.pi])
    if np.any(gaps<1e-10): raise ValueError('duplicate angles')
    if mode=='periodic': weights=(np.roll(gaps,1)+gaps)/2
    elif mode=='limited':
        cut=int(np.argmax(gaps)); order=np.roll(order,-cut-1); seq=np.unwrap(a[order]); delta=np.diff(seq)
        weights=np.empty(len(seq)); weights[0]=delta[0]/2; weights[-1]=delta[-1]/2
        if len(seq)>2: weights[1:-1]=(delta[:-1]+delta[1:])/2
    else: raise ValueError('angular_mode must be limited or periodic')
    return order,weights


def fdk(projections, views, grid, angles_deg, angular_mode='limited', chunk_depth=16):
    """Circular flat-detector FDK; limited-angle output is approximate and unnormalized."""
    projections=np.asarray(projections,dtype=float)
    if len(views)!=len(angles_deg) or projections.ndim!=3 or len(projections)!=len(views): raise ValueError('one view and angle per projection required')
    order,weights=_angular_weights(angles_deg,angular_mode)
    for view, angle, image in zip(views,angles_deg,projections):
        if tuple(image.shape)!=tuple(view.detector_shape): raise ValueError('detector shape mismatch')
        sid=np.linalg.norm(view.source[:2]); inferred=np.rad2deg(np.arctan2(view.source[0],-view.source[1]))
        if sid <= 0 or abs((inferred-angle+180)%360-180)>1e-5: raise ValueError('view angle does not match geometry')
        sdd=np.linalg.norm(view.detector_center-view.source)
        expected=circular_view(angle,sdd,sid,view.detector_shape,view.pixel_size_mm)
        if not all(np.allclose(getattr(view,key),getattr(expected,key),atol=1e-6) for key in ('source','detector_center','u_axis','v_axis')):
            raise ValueError('FDK supports only the specified circular detector orientation and geometry')
    nz,ny,nx=grid.shape; sx,sy,sz=grid.spacing_mm; ox,oy,oz=grid.origin_mm
    if chunk_depth<1: raise ValueError('chunk_depth must be positive')
    out=np.zeros(grid.shape,dtype=np.float32)
    for pos,i in enumerate(order):
        view=views[i]; image=projections[i]; h,w=image.shape
        sid=np.linalg.norm(view.source[:2]); central=view.detector_center-view.source; sdd=np.linalg.norm(central); axis=central/sdd
        vv,uu=np.meshgrid((np.arange(h)-(h-1)/2)*view.pixel_size_mm,(np.arange(w)-(w-1)/2)*view.pixel_size_mm,indexing='ij')
        weighted=image*sdd/np.sqrt(sdd*sdd+uu*uu+vv*vv)
        nfft=1 << int(np.ceil(np.log2(max(2*w,2))))
        ramp=np.abs(np.fft.rfftfreq(nfft,d=view.pixel_size_mm))
        filtered=np.fft.irfft(np.fft.rfft(weighted,n=nfft,axis=-1)*ramp,n=nfft,axis=-1)[...,:w]
        # Filtering is in detector coordinates: map the isocentre-plane derivative scale.
        filtered *= sdd/sid
        for start in range(0,nz,chunk_depth):
            z,y,x=np.meshgrid(oz+np.arange(start,min(start+chunk_depth,nz))*sz,oy+np.arange(ny)*sy,ox+np.arange(nx)*sx,indexing='ij')
            points=np.stack((x,y,z),axis=-1); uv=project_points(points,view)
            coords=np.stack((np.nan_to_num(uv[...,1],nan=-1e6),np.nan_to_num(uv[...,0],nan=-1e6)))
            sampled=map_coordinates(filtered,coords,order=1,mode='constant',cval=0)
            depth=(points-view.source)@axis
            factor=np.divide(sid,depth,out=np.zeros_like(depth),where=depth>0)**2
            out[start:start+len(z)] += (sampled*factor*weights[pos]/2).astype(np.float32)
    return out
