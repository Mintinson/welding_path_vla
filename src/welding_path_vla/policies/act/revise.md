1. 不要过度引入复杂度，要复用现用的功能，不要重复造轮子。如果有新的功能，查找有无类似的模块，将其放在该模块中，不要弄得模块到处都是。
2. 我们的代码是在 lerobot 框架中的，请尽量使用 lerobot 已有的功能和接口，而不要重复造轮子。你可以参考 .pixi/envs/default/lib/python3.12/site-packages/lerobot/scripts/lerobot_train.py 和 .pixi/envs/default/lib/python3.12/site-packages/lerobot/scripts/lerobot_rollout.py，看看它们是怎么实现的。
3. 同样的，命令行读取可以使用 draccus，而不要手写，引入太多复杂度。(也更加简洁和符合lerobot的规范和范式)