# naviai_yolo 文档导航

完整的文件架构、配置、API 和运行说明统一维护在 [项目 README](../README.md)。本页只作导航；代码审核保留独立文档。

| 要做什么 | 阅读位置 |
|---|---|
| 在业务代码中 import、实例化 Client、传参并读取结果 | [业务集成快速开始](../README.md#client-quickstart) |
| 了解项目提供哪些功能、文件如何分层 | [文件架构与职责](../README.md#architecture) |
| 配置 ROS / Python、catkin 或源码运行环境 | [环境与启动准备](../README.md#environment) |
| 修改模型、相机话题、保存开关，确认路径基准 | [配置文件与参数](../README.md#configuration) |
| 确认图像尺寸、深度单位、Mask 编码、labels 语义 | [输入与结果约定](../README.md#image-contract) |
| 调用一次 YOLO，使用数组或 RGB-D 文件 | [Service 使用与方法](../README.md#service) |
| 从相机执行目标检测任务、读取反馈、处理超时 | [Action 使用与方法](../README.md#action) |
| 本地读取 Mask、抓拍、复用相机组件或直接模型推理 | [本地方法与其他工具](../README.md#tools) |
| 查找原始数据与可视化，理解 meta.json，回放样本 | [Debug 数据保存与回放](../README.md#debug) |
| 运行回归测试、实机验收、排查常见问题 | [测试、验收与常见问题](../README.md#verification) |
| 从旧协议、脚本和配置迁移 | [旧接口迁移与当前边界](../README.md#migration) |
| 按调用链逐段检查 Service 代码 | [Service code review 路线](SERVICE_CODE_REVIEW.md) |

[Service 调用模板](../examples/seg_service_call.py)和 [Action 最小调用模板](../examples/seg_action_call.py)可作为上层业务集成起点。

Service 只执行 YOLO 推理；已有 Mask 用本地 load_mask_file()。Fdpose 的适配仍留待 fdpose_pipeline 阶段，不属于本次文档整理的功能修改。
