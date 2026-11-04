import json
from pathlib import Path
import numpy as np
import tifffile


def _projection(angle, offset=0.0):
    a = np.deg2rad(angle)
    # Orthographic camera with a modest vertical coordinate and depth component.
    return np.array([[np.cos(a), 0.0, np.sin(a), 32 + offset],
                     [0.0, 1.0, 0.12, 32],
                     [0.002 * np.sin(a), 0.0, 0.002 * np.cos(a), 1.0]], dtype=float)


def _project(P, X):
    q = P @ np.r_[X, 1.0]
    return (q[:2] / q[2]).tolist()


def make_synthetic_dataset(out_dir, seed=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = []
    groups = [f"scaffold_{i:02d}" for i in range(6)]
    for gi, group in enumerate(groups):
        difficulty = ("normal", "close", "low_disparity")[gi % 3]
        angles = [-15.0, 0.0, 15.0] if difficulty != "low_disparity" else [-0.3, 0.0, 0.3]
        Xs = [np.array([-8., -8., 5.]), np.array([8., 8., -5.])]
        if difficulty == "close": Xs = [np.array([-0.35, 0., 0.]), np.array([0.35, 0., 0.])]
        views = []
        for vi, angle in enumerate(angles):
            P = _projection(angle, gi * .1)
            xy = np.asarray([_project(P, x) for x in Xs])
            image = rng.normal(8, 1, (64, 64)).clip(0, 255).astype(np.uint8)
            mask = np.zeros((64,64), dtype=np.uint8)
            yy, xx = np.mgrid[:64,:64]
            for x,y in xy:
                image = np.maximum(image, (220*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.15**2))).astype(np.uint8))
                mask[((xx-x)**2+(yy-y)**2) <= 3**2] = 1
            stem=f"{group}_view{vi}"
            tifffile.imwrite(out/f"{stem}.tif", image)
            tifffile.imwrite(out/f"{stem}_mask.tif", mask)
            views.append({"view_id":vi,"angle_deg":angle,"projection_path":f"{stem}.tif","mask_path":f"{stem}_mask.tif","marker_xy":xy.tolist(),"P":P.tolist()})
        cases.append({"case_id":f"case_{gi:02d}","group_id":group,"difficulty":difficulty,"views":views})
    folds={"fold_0":{"train":groups[:4],"val":[groups[4]],"test":[groups[5]]}}
    manifest={"schema_version":"phase1_synthetic_v1","seed":seed,"cases":cases,"folds":folds}
    path=out/"manifest.json"; path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return path
