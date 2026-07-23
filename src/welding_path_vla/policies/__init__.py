"""Policy, training, and deployment contracts independent of robot hardware."""

from welding_path_vla.policies.base import Observation, Policy

__all__ = ["Observation", "Policy"]
