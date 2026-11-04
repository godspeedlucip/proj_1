from itertools import combinations
import numpy as np


def triangulate_dlt(points, projection_matrices):
    rows=[]
    for (u,v),P in zip(points,projection_matrices):
        P=np.asarray(P,dtype=float)
        rows.extend([u*P[2]-P[0],v*P[2]-P[1]])
    A=np.asarray(rows,dtype=float)
    norms=np.linalg.norm(A,axis=1); A=A/np.maximum(norms[:,None],1e-12)
    _,s,vt=np.linalg.svd(A,full_matrices=False)
    Xh=vt[-1]
    if not np.isfinite(Xh).all() or abs(Xh[3])<1e-10: raise ValueError("unstable triangulation (point at infinity)")
    X=Xh[:3]/Xh[3]
    if not np.isfinite(X).all(): raise ValueError("non-finite triangulation")
    return X, float(s[-2]/max(s[-1],1e-12))


def project_point(P, X):
    q=np.asarray(P,dtype=float)@np.r_[np.asarray(X,dtype=float),1.0]
    if not np.isfinite(q).all() or q[2]<=1e-10: raise ValueError("point has invalid projection depth")
    return q[:2]/q[2]


def ray_parallax_degrees(projection_matrices, X):
    """Return the largest pairwise angle between camera rays for a 3D point."""
    centers=[]; rays=[]
    for P in projection_matrices:
        M=np.asarray(P,dtype=float)[:,:3]
        if abs(np.linalg.det(M))<1e-12: return 0.0
        center=-np.linalg.solve(M,np.asarray(P,dtype=float)[:,3])
        ray=np.asarray(X,dtype=float)-center
        n=np.linalg.norm(ray)
        if not np.isfinite(n) or n<1e-12: return 0.0
        centers.append(center); rays.append(ray/n)
    angles=[]
    for i,j in combinations(range(len(rays)),2):
        angles.append(np.rad2deg(np.arccos(np.clip(np.dot(rays[i],rays[j]),-1.0,1.0))))
    return float(max(angles,default=0.0))
