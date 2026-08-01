"""π0 离线评估入口。"""

from welding_path_vla.core.config import AppConfig
from welding_path_vla.policies.pi_family.evaluation import PIEvaluationReport
from welding_path_vla.policies.pi_family.evaluation import (
    evaluate_checkpoint as family_evaluate_checkpoint,
)
from welding_path_vla.policies.pi_family.spec import PI0


def evaluate_checkpoint(config: AppConfig, checkpoint: str) -> PIEvaluationReport:
    """在任务均衡的留出 episode 上评估 π0。"""
    return family_evaluate_checkpoint(config, checkpoint, PI0)


__all__ = ["PIEvaluationReport", "evaluate_checkpoint"]
