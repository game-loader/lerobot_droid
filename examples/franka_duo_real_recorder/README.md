# Franka Duo 真机 LeRobot v3 采集器

这是一个独立的 ROS 2 真机采数工具，输出 LeRobot v3 数据集。它复用
`/home/droid/benchmark/task2_isaacsim` 的语义 topic 名称，但不依赖 Isaac
Sim。真机驱动可以直接发布这些 topic，也可以使用 ROS relay 做映射。

## 数据契约

- `action`: 17 维 `float32`，左右臂各 7 个关节目标、左右夹爪目标、spine 目标。
  不包含 base `vx/vy/wz`。
- `observation.state`: 17 维 `float32`，左右臂实测关节、左右夹爪、spine。
  不写不可获得的 world pose、底盘里程计或底盘速度，也不在采集循环中做 FK。
- `observation.images.head`: ZED Mini RGB，默认 `1280x720 @ 30 FPS`。
- `observation.images.wrist_left/right`: D405 RGB，默认 `480x270 @ 30 FPS`。
- ZED Mini 深度不放入 LeRobot RGB 视频，而写入
  `franka_duo_extras/episode_NNNNNN/`。

### D405 官方模式

RealSense D400 Series Datasheet（Revision 023，2026-03，Table 4-4）列出的
D405 USB 3.1 Gen 1 彩色流（左深度 imager 经 ISP 输出 YUY2/UYVY）模式为：
[官方 Datasheet](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf)。

| 分辨率     | 官方 FPS          |
| ---------- | ----------------- |
| `1280x720` | 5, 15, 30         |
| `848x480`  | 5, 15, 30, 60, 90 |
| `640x480`  | 5, 15, 30, 60, 90 |
| `640x360`  | 5, 15, 30, 60, 90 |
| `480x270`  | 5, 15, 30, 60, 90 |
| `424x240`  | 5, 15, 30, 60, 90 |

D405 没有独立 RGB sensor；所谓 RGB 是左侧 global-shutter depth imager 经过
ISP 产生的匹配彩色流。librealsense 给 D405 标记的默认 USB 3 profile 是
`848x480 @ 30 FPS`（USB 2 默认 `848x480 @ 10 FPS`），但本工具按比赛采集负载
将双腕默认设为官方支持的 `480x270 @ 30 FPS`。单路原始 RGB 约 11.7 MB/s，
两路相较 `848x480` 合计减少约 50 MB/s 的内存/DDS 搬运。硬件能枚举 90 FPS
不代表三相机系统应按 90 FPS 录制。

两台 D405 应尽量接不同 USB 3 控制器，不要共用低带宽 hub。现场必须用设备
枚举确认当前固件和 ROS driver 实际暴露的 profile：

```bash
rs-enumerate-devices -c
lerobot-find-cameras realsense
```

采集器不会把 D405 深度订阅进默认数据集；`wrist_left/right` 的 `record_depth`
固定为 `false`，只有 `head` 的 `record_depth` 为 `true`。

## 深度与 RGB 同步保证

ROS 回调只负责把消息放入内存缓存，深度写盘由独立线程完成。每个被采样的
head RGB 帧都会用 `header.stamp` 在深度历史中寻找最近深度帧：

1. 时间差超过 `depth_match_tolerance_ms`（默认 16 ms，小于 30 FPS 半帧周期）则记入
   `missing_rgb_frame_indices`，不会伪造或复用无时间戳的深度。
2. 匹配成功时同时写入 RGB `frame_index`、RGB stamp、depth stamp，二进制深度
   数据按固定 shape 顺序写入。写盘速度不会改变匹配关系。
3. 深度队列满时明确计入 `frames_dropped`；训练前应拒绝或检查该 episode。

因此“异步写盘”不会破坏同步；同步依据是 ROS header 时间戳，不是到达顺序。
写盘线程只改变 I/O 时机，不改变已经确定的 RGB/depth 配对。默认
`depth_every: 1` 为每个 30 FPS head RGB 保存一帧独立 depth；`1280x720`
float16 顺序 sidecar 约为 55 MB/s。若现场存储无法持续该带宽，可显式设为
`2` 降到 15 FPS；队列溢出或任一请求帧没有合格 depth 时，默认都会拒绝整个
episode，不会静默生成错位数据。只有显式传 `--allow-missing-depth` 才放宽后者。

主采样也不再按 wall-clock 重复读取 latest：每个新的 head RGB stamp 最多生成
一个样本，左右腕、实测 state 和 applied action 都从短历史中按 head stamp
近邻匹配并受独立 skew 阈值约束。LeRobot parquet 中的 `timestamp` 仍是标准的
`frame_index / fps`；所有真实 ROS 源时间写在每个 episode 的
`sample_timestamps_ns.i8`，布局记录在 `metadata.json` 中，便于训练前审计实际
频率和跨 topic skew。

数据集级 `franka_duo_extras/recording_manifest.json` 固化 action/state schema、
ROS topics、三路相机配置、同步阈值、深度 cadence、夹爪标定和编码设置。
`--resume` 会同时校验 manifest 与 LeRobot metadata，配置不一致时拒绝追加。

采集器要求三路 `CameraInfo` 在录制前到达，校验尺寸、焦距和 frame id，并在
episode 内拒绝标定变化。它同时保存内参、编码、单位和 frame id。若后续要生成
机器人基座坐标系点云，还必须在配置中提供经过标定的 head optical frame 到
robot base 的 4x4 外参（或用 TF relay 后填写该矩阵）：

```yaml
head_to_robot_base_transform:
  [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz, 0, 0, 0, 1]
```

不提供外参时，后续只能得到相机坐标系点云，不能声称是 robot-base/world 点云。

## FK 离线计算

采集时不导入 Pinocchio/URDF，避免 FK 计算抖动影响 30 Hz 采样。采集结束后，
使用实际安装和关节命名对应的 URDF 计算 sidecar：

```bash
uv run python examples/franka_duo_real_recorder/enrich_fk.py \
  --dataset-root datasets/franka_duo/franka_duo_real_v1 \
  --urdf /path/to/calibrated_fr3_duo.urdf \
  --left-joints left_fr3v2_joint1,left_fr3v2_joint2,left_fr3v2_joint3,left_fr3v2_joint4,left_fr3v2_joint5,left_fr3v2_joint6,left_fr3v2_joint7 \
  --right-joints right_fr3v2_joint1,right_fr3v2_joint2,right_fr3v2_joint3,right_fr3v2_joint4,right_fr3v2_joint5,right_fr3v2_joint6,right_fr3v2_joint7 \
  --left-frame left_fr3v2_link8 --right-frame right_fr3v2_link8
```

输出 `franka_duo_extras/fk_ee_pose_xyzw.npy`，顺序为左臂
`xyzqxqyqzqw`、右臂 `xyzqxqyqzqw`，坐标系由 URDF 和静态 mount transform
决定，默认不是 world。

## 运行

先确保控制器发布：

- `/isaac/joint_states_full` (`sensor_msgs/JointState`)：实测状态；
- `/isaac/applied_joint_commands` (`sensor_msgs/JointState`)：控制器最终应用的
  绝对关节目标。真机没有该 topic 时必须从 teleop/controller command 入口
  relay，不能用 measured state 冒充 action；
- 三路 RGB、head 深度、三路 `camera_info`。

两个 `JointState` 都必须带同一 ROS clock 域的非零 `header.stamp`，并包含左右
7 关节、左右夹爪和 `franka_spine_vertical_joint`。spine 既然属于 17 维契约，
缺失时采集器会拒绝样本，不会用 `0` 伪造。

然后：

```bash
examples/franka_duo_real_recorder/run_recorder.sh \
  --gripper-closed-rad 0.0 \
  --gripper-open-rad 0.04 \
  --auto-start --episodes 50
```

上面的 `0.0/0.04` 只是标准 Franka finger joint（单位 m）的常见示例，不是
可直接信任的比赛配置。必须分别在真机上读取完全闭合和完全打开时的
`JointState.position` 并填写两个端点。若 controller topic 使用角度或反向定义，
直接填写该 topic 的实际端点；归一化公式同时支持 `open < closed`。当前配置对
左右夹爪使用同一对端点，若两边标定不同，应先由 ROS relay 将两边转换为统一
位置契约后再采集。

交互模式下按 `r` 开始，录制中按 `s` 保存、`d` 丢弃、`q` 丢弃并退出。默认
开启 LeRobot 流式编码；NVIDIA 主机的 `rgb_vcodec: auto` 会优先选择
`h264_nvenc`，否则回退到软件编码。输出目录默认为
`datasets/franka_duo/franka_duo_real_vN/`。

每次正式训练前运行严格验收：

```bash
uv run python examples/franka_duo_real_recorder/validate_dataset.py \
  --dataset-root datasets/franka_duo/franka_duo_real_v1
```

该命令检查 LeRobot v3 schema、action/state 有限值、每路视频 episode 时长、
真实 ROS stamp 单调性与 FPS、RGB/control/depth skew、深度 cadence、文件大小和
缺失率。默认 depth 缺失率上限为 0%；不通过时退出码为 1 并列出具体 episode。
只有明确接受稀疏 depth 时才传 `--max-depth-missing-ratio` 放宽。

## DP3 点云派生集

本分支已引入 LeRobot 原生 `dp3` policy。其输入契约是固定数量点云（默认
512 点）和 `observation.state`；它复用 LeRobot Diffusion 的 UNet、processor
和 checkpoint 机制，但不直接加载 RL-100 `RL1003D` checkpoint，需要用真机
数据重新训练。

采集完成后，从已同步 ZED depth sidecar 生成纯点云 LeRobot v3 派生集：

```bash
uv run python examples/franka_duo_real_recorder/build_pointcloud_dataset.py \
  --dataset-root datasets/franka_duo/franka_duo_real_v1 \
  --output-root datasets/franka_duo/franka_duo_real_dp3_v1 \
  --num-points 512 \
  --workspace-min=-0.8,-0.8,0.0 \
  --workspace-max=0.8,0.8,1.5
```

默认要求采集时已有 `head_to_robot_base_transform`，否则必须显式传
`--extrinsics`。只有确实接受相机坐标系训练时才使用 `--allow-camera-frame`。
默认移除三路 RGB 视频，避免 DP3 训练仍解码不用的图像；`--keep-videos` 可保留。
派生集会重算 numeric stats，并在 `pointcloud_extras/` 保存每帧来源、完整标定和
SHA-256 provenance。训练参数见 `docs/source/dp3.mdx`。
