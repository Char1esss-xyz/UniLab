# UniLab — Wheelbipe V14 Flat

此精简版本基于 UniLab 最新 Manager-Based 运行时，仅保留
`Robotics-Wheelbipe-V14-Flat-v0` 对应的 MuJoCo 环境和标准 PPO 训练。
UniLab 注册任务名为 `WheelbipeV14Flat`，命令行任务名为 `wheelbipe_v14_flat`。

使用 `uv` 安装依赖：

```bash
uv sync
```

开始训练：

```bash
uv run train
```

原有的显式参数形式也可使用：

```bash
uv run train --algo ppo --task wheelbipe_v14_flat --sim mujoco
```

MuJoCo 仿真和环境 Manager 在 CPU 上运行；PPO 自动选择可用学习设备，
也可使用 `--device cpu`、`--device mps` 或 `--device cuda` 指定。
完整训练建议在 Ubuntu 上进行；macOS 可用于开发和短训练检查。
训练不需要 Isaac Lab，也不加载原 Isaac Lab 权重。

使用 MJWarp 在 CUDA 上运行仿真和 PPO：

```bash
uv sync --extra mjwarp
uv run --extra mjwarp train --sim mjwarp --device cuda:0
```

也可运行 `tools/run_mjwarp_ppo.sh`；它会启用 MJWarp 依赖，并接受
`DEVICE=cuda:0`、`MJWARP_NCONMAX`、`MJWARP_NJMAX` 等环境变量。

短训练检查：

```bash
uv run train --num-envs 16 --max-iterations 2 --device cpu algo.save_interval=1
```

训练输出默认保存至 `logs/rsl_rl_ppo/WheelbipeV14Flat/<时间>_mujoco/`，
包含解析后的 `config.yaml`、TensorBoard 记录和 `model_*.pt`。
`run_config.json` 保存策略输入和网络约定；恢复训练、播放和手动策略播放会在构建环境前
校验这些约定。旧训练目录会使用原始 `config.yaml`，因此请将配置与 checkpoint 一起保留。
可通过 `--log-root /path/to/logs` 指定输出根目录。

从 checkpoint 继续训练：

```bash
uv run train --resume-path /path/to/model_200.pt --max-iterations 1000
```

训练过程中按 `Ctrl+C` 会先保存一个带有 `_interrupt.pt` 后缀的 checkpoint，
然后再退出。下一次启动时把这个文件传给 `--resume-path` 即可继续；建议把保存间隔调小，
例如 `algo.save_interval=100`，这样即使机器断电也只会丢失最近一小段训练：

```bash
uv run train --device cuda --num-envs 1024 --max-iterations 5000 \
  algo.save_interval=100
```

`--max-iterations` 表示本次新增训练迭代数。续训恢复模型、优化器、训练进度和日志计数；
不恢复模拟器的瞬时物理状态。续训应使用与 checkpoint 相同的网络和环境配置，
例如通过命令末尾的 Hydra 参数设置 `algo.seed=1`。

播放训练好的 checkpoint 并导出 MuJoCo 视频：

```bash
uv run play \
  --checkpoint logs/rsl_rl_ppo/WheelbipeV14Flat/<训练目录>/model_20000.pt \
  --device cuda \
  --steps 200 \
  --output playback.mp4
```

MuJoCo 播放入口使用确定性策略，默认运行一个环境并输出 MP4。只做无渲染的回合统计时使用
`--render-mode none`；增加 `--num-envs` 可以同时评估多个环境。

仓库内的物理模型可直接用于训练和播放，无需额外视觉缓存。
如果本地存在可用的 V14 USD 视觉缓存，启动时会在缓存目录生成带视觉网格的场景；
视觉网格不参与碰撞，原始物理 XML 保持不变。未安装缓存或缓存引用的网格文件缺失时，
使用物理几何体显示。
当前 MuJoCo 模型只保留底盘和轮腿，云台视觉、刚体、关节和执行器均已移除。
旧 checkpoint 的策略维度仍可用于短时播放验证，但物理模型已变化，正式训练请重新开始。
MJWarp 的角速度观测现统一为机体系，与 MuJoCo 一致；修复前的 MJWarp 权重需要重新评估，
正式训练建议重新开始。
近距离查看底盘外观可使用：

```bash
uv run play \
  --checkpoint logs/rsl_rl_ppo/WheelbipeV14Flat/<训练目录>/model_20000.pt \
  --device cpu --steps 200 --output wheelbipe_full.mp4 \
  --cam-distance 2.0 --cam-elevation -15 --cam-azimuth 90
```

可选：从你自己的 Wheelbipe V14 USD 文件生成或更新本地视觉缓存。
转换器只导出当前底盘任务保留的刚体，旧缓存中的云台条目会自动忽略。
缓存目录已被 `.gitignore` 忽略，后续训练和播放会自动使用它：

```bash
WHEELBIPE_USD=/path/to/wheelbipeV14_2_base.usd
uv run --with usd-core \
  python tools/convert_wheelbipe_usd_visuals.py \
  "$WHEELBIPE_USD" \
  .unilab_assets/wheelbipe_v14_visuals
```

需要提前生成并检查视觉场景时可运行以下命令；它会输出缓存场景路径：

```bash
uv run python tools/attach_mujoco_visuals.py \
  src/unilab/assets/robots/wheelbipe_v14/robot.xml \
  .unilab_assets/wheelbipe_v14_visuals/manifest.json
```

任务参数位于 `src/unilab/conf/ppo/task/wheelbipe_v14_flat/`，
PPO 参数位于 `src/unilab/conf/ppo/config.yaml`。
机器人资产及许可证位于 `src/unilab/assets/robots/wheelbipe_v14/`。

开发检查：

```bash
make check
make test
```
