"""Constrained FDK experiment planner, discoverable by nnU-Net."""

from __future__ import annotations

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.phase3.config import Phase3Config
from nnunetv2.phase3.data_conversion import PreparedFold
from nnunetv2.phase3.planning import make_phase3_plans


class FDKnnUNetPlanner(ExperimentPlanner):
    def __init__(self, prepared: PreparedFold, config: Phase3Config):
        self.prepared = prepared
        self.config = config
        self.plans = None

    def plan_experiment(self) -> dict:
        self.plans = make_phase3_plans(self.prepared, self.config)
        return self.plans
