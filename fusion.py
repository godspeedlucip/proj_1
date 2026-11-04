from dataclasses import dataclass
from itertools import combinations, product
import numpy as np
from .geometry import triangulate_dlt, project_point, ray_parallax_degrees
from .peaks import Candidate


@dataclass
class FusionResult:
    marker_xy: list
    pair_status: str
    fusion_mode: list
    association_status: str
    diagnostics: dict
    points_3d: list | None = None
    view_pair_status: list | None = None


def _as_candidate(c):
    if isinstance(c,Candidate): return c
    if isinstance(c,dict): return Candidate(float(c["x"]),float(c["y"]),float(c["score"]))
    return Candidate(*map(float,c))


def fuse_three_views(candidate_sets, projection_matrices, config=None):
    cfg=dict(config or {})
    if len(candidate_sets)!=3 or len(projection_matrices)!=3: raise ValueError("three candidate sets and projection matrices are required")
    sets=[[_as_candidate(c) for c in s] for s in candidate_sets]
    Ps=[np.asarray(p,dtype=float) for p in projection_matrices]
    if any(p.shape!=(3,4) or not np.isfinite(p).all() for p in Ps): raise ValueError("each projection matrix must be finite 3x4")
    min_angle=float(cfg.get("min_parallax_deg",1.0)); lam=float(cfg.get("lambda_reproj",1.0)); scale=max(float(cfg.get("scale_px",2.0)),1e-9)
    eps=1e-9; valid=[]; errors=[]
    pairs=[]
    for cs in sets:
        generated=[]
        for a,b in combinations(cs,2):
            generated.append((a,b,a.score+b.score))
        generated.sort(key=lambda p:-p[2]); pairs.append(generated[:max(1,int(cfg.get("top_l",cfg.get("L",5))))])
    if all(pairs):
        for chosen in product(*pairs):
            (a0,b0,_),(a1,b1,_),(a2,b2,_)=chosen
            for swaps in product((False,True),repeat=2):
                points_by_view=[[a0,b0],[b1,a1] if swaps[0] else [a1,b1],[b2,a2] if swaps[1] else [a2,b2]]
                try:
                    xyz=[]; reproj=[]; ratios=[]
                    for k in range(2):
                        X,ratio=triangulate_dlt([(v[k].x,v[k].y) for v in points_by_view],Ps); ratios.append(ratio); xyz.append(X)
                        for vi in range(3): reproj.append(float(np.linalg.norm(project_point(Ps[vi],X)-[points_by_view[vi][k].x,points_by_view[vi][k].y])))
                    parallax=min(ray_parallax_degrees(Ps,X) for X in xyz)
                    if parallax < min_angle: raise ValueError(f"effective parallax {parallax:.4g} deg below configured threshold {min_angle:.4g} deg")
                    huber=np.where(np.asarray(reproj)<=scale,.5*np.asarray(reproj)**2/scale,np.asarray(reproj)-.5*scale)
                    score=float(np.mean([np.log(max(v.score,eps)) for pair in points_by_view for v in pair])-lam*np.mean(huber))
                    valid.append((score,points_by_view,xyz,reproj,ratios,parallax))
                except (ValueError,np.linalg.LinAlgError) as exc:
                    errors.append(str(exc))
    if valid:
        valid.sort(key=lambda z:-z[0]); best=valid[0]
        close_second=bool(len(valid)>1 and abs(best[0]-valid[1][0])<float(cfg.get("uncertain_margin",1e-3)))
        same_unordered=close_second and all(np.allclose(sorted((tuple((c.x,c.y)) for c in pa)),sorted((tuple((c.x,c.y)) for c in pb)),atol=1e-6) for pa,pb in zip(best[1],valid[1][1]))
        ambiguous=close_second and same_unordered
        uncertain=close_second and not same_unordered
        return FusionResult([[(c.x,c.y) for c in pair] for pair in best[1]],"uncertain" if uncertain else "reliable",["three_view"]*3,"ambiguous" if ambiguous else ("uncertain" if uncertain else "resolved"),{"score":best[0],"reprojection_error_px":float(np.mean(best[3])),"triangulation_condition":best[4],"effective_parallax_deg":best[5],"rejected_hypotheses":len(errors)},[x.tolist() for x in best[2]],["uncertain" if uncertain else "reliable"]*3)
    fallback=[]
    for cs in sets:
        if len(cs)<2:
            fallback.append(None)
        else:
            top=sorted(cs,key=lambda c:-c.score)[:2]; fallback.append([(c.x,c.y) for c in top])
    statuses=["invalid" if pair is None else "reliable" for pair in fallback]
    modes=[None if pair is None else "single_view_fallback" for pair in fallback]
    overall="invalid" if "invalid" in statuses else "reliable"
    return FusionResult(fallback,overall,modes,"unknown",{"reason":errors[0] if errors else "insufficient distinct candidates","candidate_counts":[len(s) for s in sets],"geometry_errors":errors[:5]},None,statuses)
