# AirVLN Scene16 前端与高清回放优化

本项目以 Liu 等人 2023 年论文《AerialVLN: Vision-and-Language Navigation for UAVs》及其官方代码为基础，针对 Scene16 单场景演示增加成功轨迹筛选、AirSim 高清离线渲染和稳定浏览器播放。

## 优化内容

- 只保留严格成功结果：终点距离不超过 20 米、落点表面方向对齐，并满足 `success_20m_with_surface=true`。
- 使用已记录的成功 trace 逐点设置 AirSim 位姿，演示渲染不改变原始终点。
- `preview_0` 以 `1280x720` 捕获，使用 H.264、8 fps 输出。
- 使用真实帧保持播放，不再使用容易造成重影和建筑扭曲的光流插帧。
- 增加 Scene16 演示构建器、视频缓存和前端回放入口。

## 当前结果

稳定池共 82 条记录。已完成 76 条高清回放；1 条因 AirSim 返回空图像而失败并剔除，另外 5 条不属于严格完成集合。已验证样片为 `1280x720`、H.264、8 fps、228 帧，渲染终点与原始记录逐坐标一致，原记录到目标距离为 15.87 米。

## 目录

```text
Model/                 CMA 策略和编码器
airsim_plugin/         AirSim 客户端/服务端集成
src/                   训练与环境代码
live_server/           演示 API、会话和浏览器前端
scripts/               评估与高清回放工具
docs/                  优化记录与复现说明
```

运行时目录、模拟器二进制、数据集、模型权重和生成视频均不进入仓库。请按照上游 AerialVLN 项目说明准备这些外部资源。

## 高清回放

```bash
python scripts/render_scene16_highres_trace.py \\
  --session-id <成功会话ID> \\
  --tool-port 30014 \\
  --camera preview_0 \\
  --fps 8
```

批量处理可使用 `--all-eligible --skip-existing`。工具会复用同一个场景连接，视频完整写入后才更新清单，支持断点续跑。

## 启动演示服务

```bash
python live_server/app.py --host 127.0.0.1 --port 18080
```

浏览器打开 `http://127.0.0.1:18080/`。模拟器和数据集路径请在本机通过环境变量配置，不要提交机器绝对路径。

## 按 README 复现优化流程

1. 克隆本仓库，并按照上游 AerialVLN 项目准备模拟器、AerialVLN/AerialVLN-S 标注和 checkpoint。这些大文件或受许可约束的资产不随本仓库分发。

2. 创建 Python 3.8 环境并安装依赖：

```bash
conda create -n AirVLN-opt python=3.8
conda activate AirVLN-opt
pip install -r requirements.txt
pip install airsim==1.7.0
```

3. 将 `config.example.env` 复制为 `.env`，填写本机数据集、checkpoint、稳定池和 manifest 路径。示例中的路径是占位符。

4. 启动 AirSim Scene16，并把 `preview_0` 配置为 1280x720。建议设置 `AIRVLN_PREVIEW_CAMERA_WIDTH=1280`、`AIRVLN_PREVIEW_CAMERA_HEIGHT=720`、`AIRVLN_PREVIEW_CAMERA_MAX_RAW_BYTES=3000000`。

5. 使用 `scripts/run_demo_server.sh` 启动演示 API，用 `GET /api/health` 检查服务，用 `GET /api/scene16/demo-builder?max_seeds_per_family=120` 查看指令目录。

6. 对已有成功会话生成一条确定性高清回放：

```bash
python scripts/render_scene16_highres_trace.py \\
  --session-id <成功会话ID> \\
  --manifest "$AIRVLN_SCENE16_PREVIEW_MANIFEST" \\
  --tool-port 30014 \\
  --skip-existing
python scripts/validate_highres_replay.py \\
  live_server/runtime/agent_sessions/<成功会话ID>/replay/replay_h264_highres_8fps.json
```

7. 单条验收通过后，再使用 `--all-eligible --skip-existing` 批量处理。渲染器复用一个 Scene16 连接，每条视频完整写入后才更新 manifest，单条失败不会中断后续任务。

如果要复现发布结果，仍需要原始成功 trace 和有许可的模拟器资源。建议同时保存 manifest、元数据 JSON、源 trace 哈希、模拟器设置哈希和输出视频 SHA256，以便审计。

## 论文引用

```bibtex
@inproceedings{liu2023aerialvln,
  title={AerialVLN: Vision-and-Language Navigation for UAVs},
  author={Liu, Shubo and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2023}
}
```
