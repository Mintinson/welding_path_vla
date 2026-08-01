"""π0.5 在项目统一策略接口中的注册入口。"""

from welding_path_vla.policies.pi_family.pipeline import PIPipeline
from welding_path_vla.policies.pi_family.spec import PI05

PI05Pipeline = PIPipeline
PIPELINE = PI05Pipeline(PI05)

__all__ = ["PIPELINE", "PI05Pipeline"]
