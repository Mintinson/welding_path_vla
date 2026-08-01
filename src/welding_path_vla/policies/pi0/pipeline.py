"""π0 在项目统一策略接口中的注册入口。"""

from welding_path_vla.policies.pi_family.pipeline import PIPipeline
from welding_path_vla.policies.pi_family.spec import PI0

PI0Pipeline = PIPipeline
PIPELINE = PI0Pipeline(PI0)

__all__ = ["PIPELINE", "PI0Pipeline"]
