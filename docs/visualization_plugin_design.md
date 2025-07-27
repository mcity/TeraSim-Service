# TeraSim 可视化插件设计方案

## 概述

本文档详细描述了TeraSim可视化插件的完整实现方案，该插件能够实时显示仿真状态，包括地图、车辆位置、交通灯状态等信息。插件采用模块化设计，与现有的TeraSim-Service API无缝集成。

## 架构设计

### 整体架构

```
TeraSim-Service API
    ↓ (配置参数)
TeraSimVisualizationPlugin
    ↓ (地图数据 + 实时状态)
独立Streamlit进程
    ↓ (Web界面)
用户浏览器
```

### 核心组件

1. **TeraSimVisualizationPlugin**: 主插件类，负责数据收集和进程管理
2. **地图数据提取器**: 从SUMO网络文件提取静态地图几何信息
3. **Streamlit可视化应用**: 独立进程，实时渲染仿真状态
4. **数据传输层**: 进程间通信机制（Queue/Redis）

## 实现细节

### 1. 可视化插件核心实现

```python
import subprocess
import multiprocessing
import logging
from pathlib import Path
import json
import time
import numpy as np
from terasim.overlay import traci
from terasim.plugins.base import BasePlugin

class TeraSimVisualizationPlugin(BasePlugin):
    def __init__(
        self,
        simulation_uuid: str,
        plugin_config: dict = None,
        redis_config: dict = None,
        viz_port: int = 8501,
        auto_start_viz: bool = True,
        viz_update_freq: int = 5,
        base_dir: str = "output"
    ):
        """
        初始化可视化插件
        
        Args:
            simulation_uuid: 仿真唯一标识符
            plugin_config: 插件配置
            redis_config: Redis配置
            viz_port: Streamlit服务端口
            auto_start_viz: 是否自动启动可视化
            viz_update_freq: 更新频率（每N步更新一次）
            base_dir: 基础目录
        """
        super().__init__(simulation_uuid, plugin_config or {}, redis_config or {})
        
        self.viz_port = viz_port
        self.auto_start_viz = auto_start_viz
        self.viz_update_freq = viz_update_freq
        self.viz_process = None
        self.step_count = 0
        
        # 数据传输队列
        self.data_queue = multiprocessing.Queue(maxsize=100)
        
        # 静态地图数据缓存
        self.static_map_data = None
        
        # 设置日志
        self.logger = self._setup_logger(base_dir)
        
    def _setup_logger(self, base_dir: str) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"viz-plugin-{self.simulation_uuid}")
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = Path(base_dir) / "visualization.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def function_after_env_start(self, simulator, ctx):
        """环境启动后的钩子函数：提取地图数据并启动可视化服务"""
        try:
            # 1. 提取静态地图数据
            self.logger.info("正在提取地图数据...")
            self.static_map_data = self._extract_map_geometry(simulator.sumo_net)
            self.logger.info(f"地图数据提取完成：{len(self.static_map_data['lanes'])}条车道")
            
            # 2. 启动可视化服务
            if self.auto_start_viz:
                self._start_streamlit_service()
                
            return True
            
        except Exception as e:
            self.logger.error(f"可视化插件启动失败: {e}")
            return False
    
    def function_after_env_step(self, simulator, ctx):
        """环境步进后的钩子函数：收集并发送动态数据"""
        self.step_count += 1
        
        # 按指定频率更新数据
        if self.step_count % self.viz_update_freq == 0:
            try:
                # 收集动态数据
                dynamic_data = self._collect_dynamic_data()
                
                # 发送数据到可视化进程
                if not self.data_queue.full():
                    self.data_queue.put(dynamic_data, block=False)
                    
            except Exception as e:
                self.logger.error(f"数据收集失败: {e}")
        
        return True
    
    def function_after_env_stop(self, simulator, ctx):
        """环境停止后的钩子函数：清理资源"""
        try:
            # 停止可视化进程
            if self.viz_process and self.viz_process.poll() is None:
                self.viz_process.terminate()
                self.viz_process.wait(timeout=5)
                self.logger.info("可视化服务已停止")
                
        except Exception as e:
            self.logger.error(f"清理资源时出错: {e}")
    
    def _extract_map_geometry(self, sumo_net):
        """从SUMO网络提取地图几何数据"""
        map_data = {
            "lanes": [],
            "junctions": [],
            "traffic_lights": [],
            "bounds": {"min_x": float('inf'), "max_x": float('-inf'), 
                      "min_y": float('inf'), "max_y": float('-inf')}
        }
        
        # 提取车道数据
        for edge in sumo_net.getEdges():
            for lane in edge.getLanes():
                lane_shape = lane.getShape()
                if lane_shape:
                    # 更新边界
                    for x, y in lane_shape:
                        map_data["bounds"]["min_x"] = min(map_data["bounds"]["min_x"], x)
                        map_data["bounds"]["max_x"] = max(map_data["bounds"]["max_x"], x)
                        map_data["bounds"]["min_y"] = min(map_data["bounds"]["min_y"], y)
                        map_data["bounds"]["max_y"] = max(map_data["bounds"]["max_y"], y)
                    
                    map_data["lanes"].append({
                        "id": lane.getID(),
                        "shape": [(float(x), float(y)) for x, y in lane_shape],
                        "width": float(lane.getWidth()),
                        "speed_limit": float(lane.getSpeed()),
                        "length": float(lane.getLength())
                    })
        
        # 提取交叉口数据
        for junction in sumo_net.getNodes():
            if junction.getType() not in ["dead_end", "rail_crossing"]:
                shape = junction.getShape()
                if shape:
                    map_data["junctions"].append({
                        "id": junction.getID(),
                        "shape": [(float(x), float(y)) for x, y in shape],
                        "position": junction.getCoord(),
                        "type": junction.getType()
                    })
        
        # 提取交通灯数据
        for tls in sumo_net.getTrafficLights():
            controlled_nodes = tls.getNodes()
            for node in controlled_nodes:
                map_data["traffic_lights"].append({
                    "id": tls.getID(),
                    "position": node.getCoord(),
                    "controlled_lanes": [conn.getFromLane().getID() for conn in tls.getConnections()]
                })
        
        return map_data
    
    def _collect_dynamic_data(self):
        """收集动态仿真数据"""
        try:
            dynamic_data = {
                "timestamp": traci.simulation.getTime(),
                "vehicles": {},
                "traffic_lights": {},
                "step": self.step_count
            }
            
            # 收集车辆数据
            for veh_id in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(veh_id)
                angle = traci.vehicle.getAngle(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                
                dynamic_data["vehicles"][veh_id] = {
                    "position": [float(x), float(y)],
                    "angle": float(angle),
                    "speed": float(speed),
                    "acceleration": float(traci.vehicle.getAcceleration(veh_id)),
                    "lane_id": traci.vehicle.getLaneID(veh_id),
                    "type": traci.vehicle.getTypeID(veh_id)
                }
            
            # 收集交通灯数据
            for tl_id in traci.trafficlight.getIDList():
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                dynamic_data["traffic_lights"][tl_id] = {
                    "state": state,
                    "phase": traci.trafficlight.getPhase(tl_id),
                    "next_switch": traci.trafficlight.getNextSwitch(tl_id)
                }
            
            return dynamic_data
            
        except Exception as e:
            self.logger.error(f"动态数据收集失败: {e}")
            return {"timestamp": time.time(), "vehicles": {}, "traffic_lights": {}, "step": self.step_count}
    
    def _start_streamlit_service(self):
        """启动Streamlit可视化服务"""
        try:
            # 确定Streamlit应用文件路径
            viz_app_path = Path(__file__).parent / "streamlit_viz_app.py"
            
            if not viz_app_path.exists():
                # 如果不存在，创建临时应用文件
                viz_app_path = self._create_temp_streamlit_app()
            
            # 启动Streamlit进程
            cmd = [
                "streamlit", "run", str(viz_app_path),
                "--server.port", str(self.viz_port),
                "--server.headless", "true",
                "--server.runOnSave", "false",
                "--", 
                "--simulation_uuid", self.simulation_uuid,
                "--data_queue_id", str(id(self.data_queue))
            ]
            
            self.viz_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.info(f"🎨 可视化界面已启动: http://localhost:{self.viz_port}")
            
            # 等待一秒确保进程启动
            time.sleep(1)
            
            if self.viz_process.poll() is not None:
                stdout, stderr = self.viz_process.communicate()
                raise RuntimeError(f"Streamlit进程启动失败: {stderr}")
                
        except Exception as e:
            self.logger.error(f"启动Streamlit服务失败: {e}")
            raise
    
    def _create_temp_streamlit_app(self):
        """创建临时Streamlit应用文件"""
        temp_app_content = '''
import streamlit as st
import plotly.graph_objects as go
import json
import time
import sys
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation_uuid", required=True)
    parser.add_argument("--data_queue_id", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    st.set_page_config(
        page_title="TeraSim 实时可视化",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 TeraSim 实时仿真可视化")
    st.write(f"仿真ID: {args.simulation_uuid}")
    
    # 创建占位符
    chart_placeholder = st.empty()
    info_placeholder = st.empty()
    
    # 模拟数据显示
    while True:
        with chart_placeholder.container():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 100, 100, 0, 0],
                y=[0, 0, 100, 100, 0],
                mode='lines',
                name='示例地图'
            ))
            fig.update_layout(
                title="等待仿真数据...",
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with info_placeholder.container():
            st.metric("仿真时间", f"{time.time():.1f}s")
            st.metric("车辆数量", "0")
        
        time.sleep(1)

if __name__ == "__main__":
    main()
'''
        
        temp_file = Path("/tmp") / f"terasim_viz_{self.simulation_uuid}.py"
        temp_file.write_text(temp_app_content)
        return temp_file
    
    def inject(self, simulator, ctx):
        """注入插件到仿真器"""
        self.ctx = ctx
        self.simulator = simulator
        
        # 使用较低优先级确保在其他插件之后执行
        priority_config = {
            "before_env": {"start": 80, "step": 80, "stop": 80},
            "after_env": {"start": 80, "step": 80, "stop": 80}
        }
        
        simulator.start_pipeline.hook(
            f"{self.plugin_name}_after_env_start", 
            self.function_after_env_start, 
            priority=priority_config["after_env"]["start"]
        )
        simulator.step_pipeline.hook(
            f"{self.plugin_name}_after_env_step", 
            self.function_after_env_step, 
            priority=priority_config["after_env"]["step"]
        )
        simulator.stop_pipeline.hook(
            f"{self.plugin_name}_after_env_stop", 
            self.function_after_env_stop, 
            priority=priority_config["after_env"]["stop"]
        )

# 插件配置
DEFAULT_VIZ_PLUGIN_CONFIG = {
    "name": "terasim_visualization_plugin",
    "priority": {
        "before_env": {"start": 80, "step": 80, "stop": 80},
        "after_env": {"start": 80, "step": 80, "stop": 80}
    }
}
```

### 2. Streamlit可视化应用实现

```python
# streamlit_viz_app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import multiprocessing
from pathlib import Path
import argparse

# 页面配置
st.set_page_config(
    page_title="TeraSim 实时可视化",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TeraSim可视化器:
    def __init__(self, simulation_uuid, data_queue_id=None):
        self.simulation_uuid = simulation_uuid
        self.data_queue_id = data_queue_id
        self.static_map_data = None
        self.current_data = None
        
    def render_map_with_vehicles(self, map_data, dynamic_data):
        """渲染地图和车辆"""
        fig = go.Figure()
        
        # 绘制车道
        if map_data and "lanes" in map_data:
            for lane in map_data["lanes"]:
                x_coords = [point[0] for point in lane["shape"]]
                y_coords = [point[1] for point in lane["shape"]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='lightgray', width=3),
                    name=f'车道 {lane["id"]}',
                    showlegend=False,
                    hovertemplate=f'车道: {lane["id"]}<br>限速: {lane.get("speed_limit", "N/A")} m/s<extra></extra>'
                ))
        
        # 绘制交叉口
        if map_data and "junctions" in map_data:
            for junction in map_data["junctions"]:
                if junction["shape"]:
                    x_coords = [point[0] for point in junction["shape"]] + [junction["shape"][0][0]]
                    y_coords = [point[1] for point in junction["shape"]] + [junction["shape"][0][1]]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill='toself',
                        fillcolor='rgba(135, 206, 235, 0.3)',
                        line=dict(color='steelblue', width=1),
                        mode='lines',
                        showlegend=False,
                        hovertemplate=f'交叉口: {junction["id"]}<extra></extra>'
                    ))
        
        # 绘制车辆
        if dynamic_data and "vehicles" in dynamic_data:
            vehicles = dynamic_data["vehicles"]
            
            if vehicles:
                vehicle_x = []
                vehicle_y = []
                vehicle_text = []
                vehicle_colors = []
                
                for veh_id, veh_data in vehicles.items():
                    x, y = veh_data["position"]
                    vehicle_x.append(x)
                    vehicle_y.append(y)
                    vehicle_text.append(f"{veh_id}<br>速度: {veh_data['speed']:.1f} m/s")
                    
                    # 根据车辆类型着色
                    if "AV" in veh_id:
                        vehicle_colors.append('red')
                    elif "BV" in veh_id:
                        vehicle_colors.append('blue')
                    else:
                        vehicle_colors.append('green')
                
                fig.add_trace(go.Scatter(
                    x=vehicle_x,
                    y=vehicle_y,
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=vehicle_colors,
                        symbol='triangle-up',
                        line=dict(width=1, color='black')
                    ),
                    text=[veh_id for veh_id in vehicles.keys()],
                    textposition="top center",
                    textfont=dict(size=8),
                    name='车辆',
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        # 绘制交通灯
        if map_data and "traffic_lights" in map_data and dynamic_data and "traffic_lights" in dynamic_data:
            tl_states = dynamic_data["traffic_lights"]
            
            for tl_data in map_data["traffic_lights"]:
                tl_id = tl_data["id"]
                x, y = tl_data["position"]
                
                # 确定交通灯颜色
                color = 'gray'
                if tl_id in tl_states:
                    state = tl_states[tl_id]["state"]
                    if 'r' in state.lower():
                        color = 'red'
                    elif 'y' in state.lower():
                        color = 'yellow'
                    elif 'g' in state.lower():
                        color = 'green'
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(size=15, color=color, symbol='circle', line=dict(width=2, color='black')),
                    name=f'交通灯 {tl_id}',
                    showlegend=False,
                    hovertemplate=f'交通灯: {tl_id}<br>状态: {tl_states.get(tl_id, {}).get("state", "未知")}<extra></extra>'
                ))
        
        # 设置布局
        fig.update_layout(
            title="TeraSim 实时仿真状态",
            xaxis_title="X 坐标 (m)",
            yaxis_title="Y 坐标 (m)",
            showlegend=True,
            hovermode='closest',
            height=600,
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        
        # 设置坐标轴比例
        if map_data and "bounds" in map_data:
            bounds = map_data["bounds"]
            fig.update_xaxes(range=[bounds["min_x"] - 50, bounds["max_x"] + 50])
            fig.update_yaxes(range=[bounds["min_y"] - 50, bounds["max_y"] + 50])
        
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def render_statistics(self, dynamic_data):
        """渲染统计信息"""
        if not dynamic_data:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "仿真时间", 
                f"{dynamic_data.get('timestamp', 0):.1f}s",
                delta=f"步数: {dynamic_data.get('step', 0)}"
            )
        
        with col2:
            vehicle_count = len(dynamic_data.get('vehicles', {}))
            st.metric("车辆数量", vehicle_count)
        
        with col3:
            if dynamic_data.get('vehicles'):
                avg_speed = np.mean([v['speed'] for v in dynamic_data['vehicles'].values()])
                st.metric("平均速度", f"{avg_speed:.1f} m/s")
            else:
                st.metric("平均速度", "0.0 m/s")
        
        with col4:
            tl_count = len(dynamic_data.get('traffic_lights', {}))
            st.metric("交通灯数量", tl_count)

def main():
    st.title("🚗 TeraSim 实时仿真可视化")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation_uuid", default="demo")
    parser.add_argument("--data_queue_id", default=None)
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = TeraSim可视化器(args.simulation_uuid, args.data_queue_id)
    
    # 侧边栏控制
    st.sidebar.header("仿真控制")
    st.sidebar.write(f"仿真ID: {args.simulation_uuid}")
    
    auto_refresh = st.sidebar.checkbox("自动刷新", value=True)
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", 0.5, 5.0, 1.0, 0.5)
    
    # 主要内容区域
    chart_placeholder = st.empty()
    stats_placeholder = st.empty()
    details_placeholder = st.empty()
    
    # 数据显示循环
    if auto_refresh:
        while True:
            # 模拟数据（实际应用中应从队列获取）
            current_time = time.time()
            sample_map_data = {
                "lanes": [
                    {"id": "lane_1", "shape": [[0, 0], [100, 0]], "speed_limit": 13.89, "width": 3.5}
                ],
                "bounds": {"min_x": 0, "max_x": 100, "min_y": -10, "max_y": 10}
            }
            
            sample_dynamic_data = {
                "timestamp": current_time,
                "step": int(current_time) % 1000,
                "vehicles": {
                    "AV_1": {
                        "position": [50 + 20 * np.sin(current_time * 0.1), 0],
                        "speed": 15.0,
                        "angle": 90
                    }
                },
                "traffic_lights": {}
            }
            
            # 渲染图表
            with chart_placeholder.container():
                fig = visualizer.render_map_with_vehicles(sample_map_data, sample_dynamic_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # 渲染统计信息
            with stats_placeholder.container():
                visualizer.render_statistics(sample_dynamic_data)
            
            # 渲染详细信息
            with details_placeholder.container():
                with st.expander("详细数据", expanded=False):
                    st.json(sample_dynamic_data)
            
            time.sleep(refresh_interval)
    else:
        st.info("点击侧边栏的'自动刷新'开始实时显示")

if __name__ == "__main__":
    main()
```

### 3. API集成方案

在现有的TeraSim-Service API中添加可视化支持：

```python
# 在 terasim_service/api.py 中添加

@app.post("/start_simulation", tags=["simulations"], summary="Start a new TeraSim simulation")
async def start_simulation(
    config: Annotated[SimulationConfig, Field(description="TeraSim simulation configuration")],
    enable_visualization: bool = False,  # 新增参数
    viz_port: int = 8501,               # 新增参数
    viz_update_freq: int = 5            # 新增参数
):
    """
    启动新的TeraSim仿真，支持可选的实时可视化
    
    Args:
        config: 仿真配置
        enable_visualization: 是否启用实时可视化
        viz_port: 可视化服务端口
        viz_update_freq: 可视化更新频率（步数）
    """
    config_data = load_config(config.config_file)
    simulation_id = str(uuid.uuid4())

    # 修改 run_simulation_process 调用
    process = Process(
        target=run_simulation_process,
        args=(simulation_id, config_data, config.auto_run, enable_visualization, viz_port, viz_update_freq),
    )
    process.start()

    response = {"simulation_id": simulation_id, "message": "Simulation started"}
    
    if enable_visualization:
        response["visualization_url"] = f"http://localhost:{viz_port}"
        response["message"] += " with visualization"

    return response

# 修改 run_simulation_task 函数
async def run_simulation_task(
    simulation_id: str, 
    config: dict, 
    auto_run: bool, 
    enable_visualization: bool = False,
    viz_port: int = 8501,
    viz_update_freq: int = 5
):
    try:
        # ... 现有代码 ...
        
        # 创建插件列表
        plugins = []
        
        # 添加协同仿真插件
        terasim_cosim_plugin = TeraSimCoSimPlugin(
            simulation_uuid=simulation_id, 
            plugin_config=DEFAULT_COSIM_PLUGIN_CONFIG,
            base_dir=str(base_dir),
            auto_run=auto_run, 
        )
        plugins.append(terasim_cosim_plugin)
        
        # 添加可视化插件（如果启用）
        if enable_visualization:
            from terasim_service.plugins.visualization_plugin import TeraSimVisualizationPlugin
            
            viz_plugin = TeraSimVisualizationPlugin(
                simulation_uuid=simulation_id,
                viz_port=viz_port,
                auto_start_viz=True,
                viz_update_freq=viz_update_freq,
                base_dir=str(base_dir)
            )
            plugins.append(viz_plugin)
        
        # 注入所有插件
        for plugin in plugins:
            plugin.inject(sim, {})
        
        # 运行仿真
        await asyncio.get_event_loop().run_in_executor(executor, sim.run)
        
        # ... 现有代码 ...
        
    except Exception as e:
        logger.exception(f"Simulation {simulation_id} failed: {str(e)}")
        # ... 错误处理 ...
```

## 使用示例

### 1. 启动带可视化的仿真

```bash
# HTTP请求示例
POST http://localhost:8000/start_simulation
Content-Type: application/json

{
    "config_file": "/path/to/simulation_config.yaml",
    "auto_run": false,
    "enable_visualization": true,
    "viz_port": 8501,
    "viz_update_freq": 10
}
```

### 2. 响应示例

```json
{
    "simulation_id": "abc-123-def",
    "message": "Simulation started with visualization",
    "visualization_url": "http://localhost:8501"
}
```

### 3. Python客户端示例

```python
import requests

# 启动仿真
response = requests.post("http://localhost:8000/start_simulation", json={
    "config_file": "./config.yaml",
    "auto_run": False,
    "enable_visualization": True,
    "viz_port": 8502
})

simulation_id = response.json()["simulation_id"]
viz_url = response.json()["visualization_url"]

print(f"仿真已启动，ID: {simulation_id}")
print(f"可视化界面: {viz_url}")

# 控制仿真
while True:
    input("按回车执行一步...")
    requests.post(f"http://localhost:8000/simulation_tick/{simulation_id}")
```

## 技术特性

### 1. 性能优化
- **静态数据缓存**: 地图数据仅在启动时提取一次
- **增量更新**: 仅传输动态变化的数据
- **可配置频率**: 支持调整更新频率以平衡性能和实时性

### 2. 坐标系处理
- **SUMO坐标系**: 直接使用SUMO内部坐标，无需转换
- **自动缩放**: 根据地图边界自动设置视图范围
- **等比例显示**: 确保地图比例正确

### 3. 容错机制
- **进程隔离**: 可视化进程崩溃不影响仿真
- **优雅降级**: 可视化失败时仿真继续运行
- **资源清理**: 自动清理临时文件和进程

## 部署注意事项

### 1. 依赖要求
```bash
pip install streamlit plotly pandas numpy
```

### 2. 端口管理
- 默认端口8501，可配置
- 支持端口冲突检测和自动分配

### 3. 安全考虑
- 仅监听本地接口
- 可配置访问权限

## 扩展可能

1. **3D可视化**: 使用Three.js或Plotly 3D
2. **视频录制**: 自动生成仿真视频
3. **数据导出**: 支持轨迹数据导出
4. **多仿真支持**: 同时可视化多个仿真实例
5. **实时分析**: 集成性能指标计算和显示

---

**注意**: 本文档提供了完整的实现方案，可以直接用于开发。插件采用模块化设计，易于维护和扩展。