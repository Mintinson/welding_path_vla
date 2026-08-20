# ADR 0001：使用单一 Pixi workspace 和多用途环境

状态：已接受

## 背景

笔记本负责仿真采集和部署，服务器负责 GPU 训练；MuJoCo、TorchCodec、CUDA 和真机依赖并不
完全兼容，但共享代码必须使用同一组已锁定版本。

## 决策

项目只维护一个 `pyproject.toml` 和 `pixi.lock`，按用途组合 `sim`、`data`、`real`、`train`、
`deploy`、`policy-sim` 和 `dev` 环境。源码以 editable package 安装到每个环境。

## 后果

环境边界清晰且共享依赖不会静默漂移；代价是首次安装多个环境需要额外空间。服务器只同步源码和
锁文件，不同步 `.pixi/`。
