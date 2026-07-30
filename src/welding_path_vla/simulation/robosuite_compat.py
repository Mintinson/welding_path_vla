"""集中导入项目实际使用的 robosuite 接口。

robosuite 1.5.1 在包导入时会探测 ``robosuite_models``、GR1 的 Mink
控制器和私有宏文件。Elfin5-Pro 不依赖这些可选组件，因此只静默导入阶段
产生的提示；真正的导入异常仍会原样抛出。
"""

import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

previous_logging_level = logging.root.manager.disable
logging.disable(logging.WARNING)
try:
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        from robosuite import make
        from robosuite.environments.base import MujocoEnv, register_env
        from robosuite.models.arenas import Arena
        from robosuite.models.objects import MujocoObject
        from robosuite.models.robots import RobotModel
        from robosuite.models.tasks import Task
        from robosuite.utils.mjcf_utils import array_to_string
        from robosuite.utils.observables import Observable, sensor
finally:
    logging.disable(previous_logging_level)

__all__ = [
    "Arena",
    "MujocoEnv",
    "MujocoObject",
    "Observable",
    "RobotModel",
    "Task",
    "array_to_string",
    "make",
    "register_env",
    "sensor",
]
