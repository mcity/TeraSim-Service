# TeraSim 可视化功能使用指南

## 概述

TeraSim可视化功能已集成到CoSim插件中，提供了实时的仿真状态显示，包括：
- 🗺️ 地图布局（车道、交叉口）
- 🚗 车辆位置和状态（AV、BV、其他车辆）
- 🚶 行人（VRU）位置
- 🚦 交通灯状态
- 🚧 施工区域

## 快速开始

### 1. 启动带可视化的仿真

```python
import requests

# 启动仿真并启用可视化
response = requests.post("http://localhost:8000/start_simulation", 
    json={
        "config_file": "./config.yaml",
        "auto_run": False  # 或 True
    }, 
    params={
        "enable_viz": True,      # 启用可视化
        "viz_port": 8501,        # Streamlit端口
        "viz_update_freq": 5     # 每5步更新一次
    }
)

result = response.json()
print(f"Simulation ID: {result['simulation_id']}")
print(f"Visualization URL: {result['visualization_url']}")
```

### 2. 访问可视化界面

在浏览器中打开返回的URL（默认 http://localhost:8501）

## API参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_viz` | bool | False | 是否启用可视化 |
| `viz_port` | int | 8501 | Streamlit服务端口 |
| `viz_update_freq` | int | 5 | 可视化更新频率（仿真步数） |

## 可视化界面功能

### 控制面板（侧边栏）
- **Auto Refresh**: 自动刷新开关
- **Refresh Interval**: 刷新间隔（0.1-2秒）
- **Show Vehicle Labels**: 显示车辆ID标签
- **Show Traffic Lights**: 显示交通灯
- **Show Construction Zones**: 显示施工区域

### 主界面
- **地图视图**: 实时显示仿真场景
- **统计信息**: 仿真时间、车辆数、平均速度等
- **状态信息**: 当前仿真状态和最后更新时间

## 车辆颜色说明
- 🔴 红色三角形：AV（自动驾驶车辆）
- 🔵 蓝色三角形：BV（背景车辆）
- 🟢 绿色三角形：其他车辆
- 🟠 橙色圆形：VRU（行人/骑行者）

## 使用示例

### 示例1：手动控制模式
```bash
# 运行测试脚本
python test_visualization.py
# 选择 1 (Manual control)
```

### 示例2：自动运行模式
```bash
# 运行测试脚本
python test_visualization.py
# 选择 2 (Auto-run mode)
```

### 示例3：使用不同端口
```python
# 如果8501端口被占用，使用其他端口
response = requests.post("http://localhost:8000/start_simulation", 
    json={"config_file": "./config.yaml", "auto_run": True}, 
    params={"enable_viz": True, "viz_port": 8502}
)
```

## 技术细节

### 数据流
1. CoSim插件在启动时提取地图数据（如果启用可视化）
2. CoSim插件每步收集仿真数据并写入Redis
3. Streamlit应用从Redis读取数据并实时渲染

### 架构优势
- **集成设计**：可视化作为CoSim插件的可选功能
- **向后兼容**：不启用时行为与原来完全一致
- **资源优化**：复用现有数据收集机制

### 性能优化
- 地图数据只在启动时提取一次
- 可配置更新频率以平衡性能和实时性
- 使用Plotly优化渲染性能

## 故障排除

### Streamlit无法启动
- 检查端口是否被占用
- 确保已安装依赖：`pip install streamlit plotly`

### 无法看到车辆
- 确认仿真正在运行
- 检查Redis连接是否正常
- 查看浏览器控制台错误信息

### 性能问题
- 增大`viz_update_freq`值减少更新频率
- 关闭不需要的显示选项
- 使用更小的仿真场景测试