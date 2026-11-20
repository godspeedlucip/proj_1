"""Behavior tests for the Phase 1 truth to Phase 2 dataset bridge."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

from get_data.build import build_from_phase1_manifest
from tools.marker_guided.validate_dataset import validate_prediction_package


class GroundTruthDatasetTests(unittest.TestCase):
    def test_handoff_reports_progress_without_rereading_projections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = []
            for view_id in (0, 3, 6):
                projection = root / f"image_{view_id}.tif"
                mask = root / f"mask_{view_id}.tif"
                tifffile.imwrite(projection, np.ones((8, 8), dtype=np.float32))
                tifffile.imwrite(mask, np.zeros((8, 8), dtype=np.uint8))
                views.append({"view_id": view_id, "angle_deg": float(view_id),
                              "projection_path": str(projection), "mask_path": str(mask),
                              "marker_xy": [[1.0, 1.0], [6.0, 6.0]]})
            source = root / "source.json"
            source.write_text(json.dumps({"schema_version": "phase1_real_v1",
                "folds": {"fold_0": {"train": ["g0"], "val": [], "test": []}},
                "cases": [{"case_id": "case0", "group_id": "g0",
                           "metadata": {"geometry_id": "geo"}, "views": views}]}))
            output = io.StringIO()
            real_imread = tifffile.imread
            with patch.object(tifffile, "imread", wraps=real_imread) as reads, redirect_stdout(output):
                build_from_phase1_manifest(source, root / "handoff", root / "dataset",
                                           metadata_only=True)
            self.assertIn("Indexing", output.getvalue())
            self.assertIn("Validating", output.getvalue())
            for view in views:
                image_reads = sum(Path(call.args[0]) == Path(view["projection_path"])
                                  for call in reads.call_args_list)
                self.assertEqual(image_reads, 1)

    def test_builds_channels_from_original_tiffs_and_truth_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            for split, group in (("train", "g0"), ("val", "g1"), ("test", "g2")):
                views = []
                for view_id in (0, 3, 6):
                    image = np.full((16, 16), 7 + view_id, dtype=np.uint16)
                    label = np.zeros((16, 16), dtype=np.uint8)
                    label[6:10, 6:10] = 1
                    projection = root / "stacked" / group / f"{view_id:03d}.tif"
                    mask = root / "stacked" / group / f"{view_id:03d}_label.tif"
                    projection.parent.mkdir(parents=True, exist_ok=True)
                    tifffile.imwrite(projection, image)
                    tifffile.imwrite(mask, label)
                    views.append({"view_id": view_id, "angle_deg": view_id * 10.0,
                                  "projection_path": str(projection), "mask_path": str(mask),
                                  "marker_xy": [[3.0, 4.0], [12.0, 11.0]]})
                cases.append({"case_id": f"case_{group}", "group_id": group, "views": views,
                              "metadata": {"geometry_id": "geo_v1", "source_split": split}})
            phase1_path = root / "phase1" / "manifest.json"
            phase1_path.parent.mkdir()
            phase1_path.write_text(json.dumps({"schema_version": "phase1_real_v1", "cases": cases,
                "folds": {"fold_0": {"train": ["g0"], "val": ["g1"], "test": ["g2"]}}}))

            result = build_from_phase1_manifest(phase1_path, root / "handoff", root / "Dataset501_Test",
                                                dataset_id=501, fold_id="fold_0")
            manifest = root / "handoff" / "manifest.json"
            package = root / "handoff" / "package.json"
            validated = validate_prediction_package(manifest, package, "fold_0")
            self.assertEqual(len(validated["joined"]), 9)
            self.assertEqual(result["sample_count"], 9)
            self.assertEqual(json.loads(package.read_text())["source_registry"][0]["fit_group_ids"], [])
            self.assertEqual(json.loads(package.read_text())["oracle_source"], "projected_stent_marker_geometry")
            for split, group, directory_name in (("train", "g0", "imagesTr"),
                                                  ("val", "g1", "imagesTr"),
                                                  ("test", "g2", "imagesTs")):
                stem = f"case_{group}_v0"
                image = tifffile.imread(root / "Dataset501_Test" / directory_name / f"{stem}_0000.tif")
                point = tifffile.imread(root / "Dataset501_Test" / directory_name / f"{stem}_0001.tif")
                validity = tifffile.imread(root / "Dataset501_Test" / directory_name / f"{stem}_0003.tif")
                self.assertEqual(int(image[0, 0]), 7)
                self.assertGreater(float(point[4, 3]), 0.99)
                self.assertTrue(np.all(validity == 1))
                if split != "test":
                    label = tifffile.imread(root / "Dataset501_Test" / "labelsTr" / f"{stem}.tif")
                    self.assertEqual(int(label[7, 7]), 1)
            self.assertEqual(len(json.loads((root / "Dataset501_Test" / "splits_final.json").read_text())[0]["train"]), 3)

    def test_rejects_missing_truth_instead_of_silently_zeroing_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = root / "manifest.json"
            phase1.write_text(json.dumps({"schema_version": "phase1_real_v1",
                "folds": {"fold_0": {"train": ["g0"], "val": [], "test": []}},
                "cases": [{"case_id": "x", "group_id": "g0", "metadata": {"geometry_id": "geo"},
                           "views": [{"view_id": 0, "marker_xy": None}]}]}))
            with self.assertRaisesRegex(ValueError, "exactly three|marker_xy"):
                build_from_phase1_manifest(phase1, root / "handoff", root / "dataset",
                                           dataset_id=501, fold_id="fold_0")
            self.assertFalse((root / "dataset").exists())


if __name__ == "__main__":
    unittest.main()
