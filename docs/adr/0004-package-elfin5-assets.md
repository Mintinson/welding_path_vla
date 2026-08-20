# ADR 0004：Elfin5 资产随 Python 包分发

状态：已接受

## 背景

模型若依赖工作目录或临时 `repo/`，笔记本和服务器会加载不同版本，安装后也无法可靠定位资源。

## 决策

Elfin5 URDF、MJCF、视觉 mesh 和焊接工具资产位于
`src/welding_path_vla/assets/elfin5/`，通过包资源定位。视觉 STL 与低复杂度碰撞代理保持分离；
来源场景只用于溯源，不参与采集。

## 后果

任意工作目录和 editable/installed package 都使用同一机器人修订；更新资产时必须同时验证
TCP、碰撞代理、相机遮挡和 package data。
