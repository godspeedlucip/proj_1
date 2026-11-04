import numpy as np

from phase_1.marker_pipeline.peaks import decode_heatmap
from phase_1.marker_pipeline.fusion import fuse_three_views


def _geometry():
    # Three pinhole cameras translated horizontally; exact points at z=10.
    Ps=[]
    for cx in (-2.,0.,2.):
        Ps.append(np.array([[10.,0.,cx,0.],[0.,10.,0.,0.],[0.,0.,1.,0.]]))
    points=[np.array([-2.,-1.,10.]),np.array([2.,1.,10.])]
    sets=[]
    for P in Ps:
        candidates=[]
        for X in points:
            q=P@np.r_[X,1.]; candidates.append((q[0]/q[2],q[1]/q[2],.9))
        sets.append(candidates)
    return sets,Ps,points


def test_decode_heatmap_keeps_distinct_peaks():
    h=np.zeros((20,20)); h[4,5]=.9; h[15,14]=.8
    assert len(decode_heatmap(h,{"top_k":10,"nms_radius":2})) == 2


def test_decode_heatmap_deduplicates_flat_maximum_plateau():
    h=np.full((12,12),.7)
    assert len(decode_heatmap(h,{"top_k":10,"nms_radius":2})) == 1


def test_fusion_is_invariant_to_view_candidate_order():
    sets,Ps,_=_geometry()
    a=fuse_three_views(sets,Ps,{"min_parallax_deg":0.0})
    sets[1]=sets[1][::-1]; sets[2]=sets[2][::-1]
    b=fuse_three_views(sets,Ps,{"min_parallax_deg":0.0})
    assert a.pair_status == "reliable" and b.pair_status == "reliable"
    for va,vb in zip(a.marker_xy,b.marker_xy):
        assert np.allclose(sorted(map(tuple,va)), sorted(map(tuple,vb)), atol=1e-4)


def test_fusion_falls_back_or_marks_invalid_with_too_few_candidates():
    sets,Ps,_=_geometry()
    fallback=fuse_three_views(sets,Ps,{"min_parallax_deg":180.0})
    assert fallback.pair_status == "reliable"
    assert set(fallback.fusion_mode)=={"single_view_fallback"}
    invalid=fuse_three_views([[sets[0][0]],sets[1],sets[2]],Ps,{})
    assert invalid.pair_status == "invalid"
    assert invalid.marker_xy[1] is not None and invalid.marker_xy[2] is not None
    assert invalid.view_pair_status == ["invalid", "reliable", "reliable"]


def test_default_threshold_falls_back_for_genuinely_low_parallax(tmp_path):
    from phase_1.marker_pipeline.synthetic import make_synthetic_dataset
    from phase_1.marker_pipeline.data import load_manifest
    manifest=make_synthetic_dataset(tmp_path,seed=5)
    case=next(c for c in load_manifest(manifest) if c.difficulty=="low_disparity")
    sets=[[(float(x),float(y),.9) for x,y in v.marker_xy] for v in case.views]
    result=fuse_three_views(sets,[v.P for v in case.views],{})
    assert result.pair_status == "reliable"
    assert result.fusion_mode == ["single_view_fallback"]*3


def test_nearby_pair_never_reuses_one_candidate():
    sets,Ps,_=_geometry()
    sets[0]=[sets[0][0]]
    result=fuse_three_views(sets,Ps,{})
    assert result.pair_status == "invalid"
