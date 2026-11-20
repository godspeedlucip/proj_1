# 共享 GPU 指引

本目录是独立 Git 仓库。使用 GPU 前请读取并遵循 `/data1/liuyang/stent/AGENTS.md` 中的检测、权限、选卡、长任务监控与续训流程。Phase 2 的数据、逐轮真实帧分割、TensorBoard 和训练命令见 `docs/real-epoch-monitoring.md`。

当前连续投影回归 U-Net 的训练与 TensorBoard 命令见 `regression_unet/README.md`。给用户的命令必须可直接运行、使用绝对路径，不依赖 `export`、`PYTHONPATH`、`CUDA_VISIBLE_DEVICES` 或 shell 变量；通过 `regression_unet/launch.py` 和 `--device cuda:N` 指定运行方式。

执行 Git 提交和推送时，遵循 `/data1/liuyang/stent/AGENTS.md` 中的「Git 提交与推送」流程。
