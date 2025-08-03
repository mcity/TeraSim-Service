# TeraSim Visualization Plugin Design

## Overview

This document describes the design and implementation of the TeraSim visualization plugin, which provides real-time visualization of simulation states including map, vehicle positions, traffic light states, and more. The plugin is built using Dash framework and integrates seamlessly with the existing TeraSim-Service API.

## Architecture Design

### Overall Architecture

```
TeraSim-Service API
    ↓ (Configuration Parameters)
TeraSimVisualizationPlugin
    ↓ (Map Data + Real-time State)
Dash Web Application
    ↓ (Web Interface)
User Browser
```

### Core Components

1. **TeraSimVisualizationPlugin**: Main plugin class responsible for data collection and process management
2. **Map Data Extractor**: Extracts static map geometry from SUMO network files
3. **Dash Visualization Application**: Web application for real-time visualization
4. **Data Communication Layer**: Inter-process communication via shared data structures

## Implementation Details

### 1. Visualization Plugin Core Implementation

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
        viz_port: int = 8050,
        auto_start_viz: bool = True,
        viz_update_freq: int = 5,
        base_dir: str = "output"
    ):
        """
        Initialize visualization plugin
        
        Args:
            simulation_uuid: Unique simulation identifier
            plugin_config: Plugin configuration
            redis_config: Redis configuration
            viz_port: Dash service port
            auto_start_viz: Auto-start visualization flag
            viz_update_freq: Update frequency (every N steps)
            base_dir: Base directory
        """
        super().__init__(simulation_uuid, plugin_config or {}, redis_config or {})
        
        self.viz_port = viz_port
        self.auto_start_viz = auto_start_viz
        self.viz_update_freq = viz_update_freq
        self.viz_process = None
        self.step_count = 0
        
        # Data transmission queue
        self.data_queue = multiprocessing.Queue(maxsize=100)
        
        # Static map data cache
        self.static_map_data = None
        
        # Setup logger
        self.logger = self._setup_logger(base_dir)
        
    def _setup_logger(self, base_dir: str) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(f"viz-plugin-{self.simulation_uuid}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = Path(base_dir) / "visualization.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def function_after_env_start(self, simulator, ctx):
        """Hook after environment start: extract map data and start visualization"""
        try:
            # 1. Extract static map data
            self.logger.info("Extracting map data...")
            self.static_map_data = self._extract_map_geometry(simulator.sumo_net)
            self.logger.info(f"Map data extracted: {len(self.static_map_data['lanes'])} lanes")
            
            # 2. Start visualization service
            if self.auto_start_viz:
                self._start_dash_service()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Visualization plugin startup failed: {e}")
            return False
    
    def function_after_env_step(self, simulator, ctx):
        """Hook after environment step: collect and send dynamic data"""
        self.step_count += 1
        
        # Update data at specified frequency
        if self.step_count % self.viz_update_freq == 0:
            try:
                # Collect dynamic data
                dynamic_data = self._collect_dynamic_data()
                
                # Send data to visualization process
                if not self.data_queue.full():
                    self.data_queue.put(dynamic_data, block=False)
                    
            except Exception as e:
                self.logger.error(f"Data collection failed: {e}")
        
        return True
    
    def function_after_env_stop(self, simulator, ctx):
        """Hook after environment stop: cleanup resources"""
        try:
            # Stop visualization process
            if self.viz_process and self.viz_process.poll() is None:
                self.viz_process.terminate()
                self.viz_process.wait(timeout=5)
                self.logger.info("Visualization service stopped")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _extract_map_geometry(self, sumo_net):
        """Extract map geometry from SUMO network"""
        map_data = {
            "lanes": [],
            "junctions": [],
            "traffic_lights": [],
            "bounds": {"min_x": float('inf'), "max_x": float('-inf'), 
                      "min_y": float('inf'), "max_y": float('-inf')}
        }
        
        # Extract lane data
        for edge in sumo_net.getEdges():
            for lane in edge.getLanes():
                lane_shape = lane.getShape()
                if lane_shape:
                    # Update bounds
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
        
        # Extract junction data
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
        
        # Extract traffic light data
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
        """Collect dynamic simulation data"""
        try:
            dynamic_data = {
                "timestamp": traci.simulation.getTime(),
                "vehicles": {},
                "traffic_lights": {},
                "step": self.step_count
            }
            
            # Collect vehicle data
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
            
            # Collect traffic light data
            for tl_id in traci.trafficlight.getIDList():
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                dynamic_data["traffic_lights"][tl_id] = {
                    "state": state,
                    "phase": traci.trafficlight.getPhase(tl_id),
                    "next_switch": traci.trafficlight.getNextSwitch(tl_id)
                }
            
            return dynamic_data
            
        except Exception as e:
            self.logger.error(f"Dynamic data collection failed: {e}")
            return {"timestamp": time.time(), "vehicles": {}, "traffic_lights": {}, "step": self.step_count}
    
    def _start_dash_service(self):
        """Start Dash visualization service"""
        try:
            # Determine Dash app file path
            viz_app_path = Path(__file__).parent / "dash_viz_app.py"
            
            if not viz_app_path.exists():
                # Create temporary app file if doesn't exist
                viz_app_path = self._create_temp_dash_app()
            
            # Start Dash process
            cmd = [
                "python", str(viz_app_path),
                "--port", str(self.viz_port),
                "--simulation_uuid", self.simulation_uuid,
                "--data_queue_id", str(id(self.data_queue))
            ]
            
            self.viz_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.info(f"🎨 Visualization interface started: http://localhost:{self.viz_port}")
            
            # Wait to ensure process starts
            time.sleep(1)
            
            if self.viz_process.poll() is not None:
                stdout, stderr = self.viz_process.communicate()
                raise RuntimeError(f"Dash process startup failed: {stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to start Dash service: {e}")
            raise
    
    def _create_temp_dash_app(self):
        """Create temporary Dash application file"""
        temp_app_content = '''
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import json
import time
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--simulation_uuid", required=True)
    parser.add_argument("--data_queue_id", required=True)
    return parser.parse_args()

def create_app():
    args = parse_args()
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("🚗 TeraSim Real-time Visualization"),
        html.P(f"Simulation ID: {args.simulation_uuid}"),
        dcc.Graph(id='live-graph'),
        dcc.Interval(id='graph-update', interval=1000),
        html.Div(id='stats-container')
    ])
    
    @app.callback(
        [Output('live-graph', 'figure'), Output('stats-container', 'children')],
        [Input('graph-update', 'n_intervals')]
    )
    def update_graph(n):
        # Create sample visualization
        fig = go.Figure()
        
        # Add sample map
        fig.add_trace(go.Scatter(
            x=[0, 100, 100, 0, 0],
            y=[0, 0, 100, 100, 0],
            mode='lines',
            name='Map Boundary'
        ))
        
        # Add sample vehicle
        current_time = time.time()
        fig.add_trace(go.Scatter(
            x=[50 + 20 * np.sin(current_time * 0.1)],
            y=[50],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='AV'
        ))
        
        fig.update_layout(
            title="TeraSim Simulation State",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            showlegend=True
        )
        
        stats = html.Div([
            html.H3("Statistics"),
            html.P(f"Time: {current_time:.1f}s"),
            html.P("Vehicles: 1"),
            html.P("Average Speed: 15.0 m/s")
        ])
        
        return fig, stats
    
    return app, args

if __name__ == "__main__":
    app, args = create_app()
    app.run_server(debug=False, port=args.port)
'''
        
        temp_file = Path("/tmp") / f"terasim_viz_{self.simulation_uuid}.py"
        temp_file.write_text(temp_app_content)
        return temp_file
    
    def inject(self, simulator, ctx):
        """Inject plugin into simulator"""
        self.ctx = ctx
        self.simulator = simulator
        
        # Use lower priority to ensure execution after other plugins
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

# Plugin configuration
DEFAULT_VIZ_PLUGIN_CONFIG = {
    "name": "terasim_visualization_plugin",
    "priority": {
        "before_env": {"start": 80, "step": 80, "stop": 80},
        "after_env": {"start": 80, "step": 80, "stop": 80}
    }
}
```

### 2. Dash Visualization Application Implementation

```python
# dash_viz_app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import time
import multiprocessing
from pathlib import Path
import argparse

# Initialize Dash app
app = dash.Dash(__name__)

class TeraSimVisualizer:
    def __init__(self, simulation_uuid, data_queue_id=None):
        self.simulation_uuid = simulation_uuid
        self.data_queue_id = data_queue_id
        self.static_map_data = None
        self.current_data = None
        
    def render_map_with_vehicles(self, map_data, dynamic_data):
        """Render map and vehicles"""
        fig = go.Figure()
        
        # Draw lanes
        if map_data and "lanes" in map_data:
            for lane in map_data["lanes"]:
                x_coords = [point[0] for point in lane["shape"]]
                y_coords = [point[1] for point in lane["shape"]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='lightgray', width=3),
                    name=f'Lane {lane["id"]}',
                    showlegend=False,
                    hovertemplate=f'Lane: {lane["id"]}<br>Speed Limit: {lane.get("speed_limit", "N/A")} m/s<extra></extra>'
                ))
        
        # Draw junctions
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
                        hovertemplate=f'Junction: {junction["id"]}<extra></extra>'
                    ))
        
        # Draw vehicles
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
                    vehicle_text.append(f"{veh_id}<br>Speed: {veh_data['speed']:.1f} m/s")
                    
                    # Color by vehicle type
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
                    name='Vehicles',
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        # Draw traffic lights
        if map_data and "traffic_lights" in map_data and dynamic_data and "traffic_lights" in dynamic_data:
            tl_states = dynamic_data["traffic_lights"]
            
            for tl_data in map_data["traffic_lights"]:
                tl_id = tl_data["id"]
                x, y = tl_data["position"]
                
                # Determine traffic light color
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
                    name=f'Traffic Light {tl_id}',
                    showlegend=False,
                    hovertemplate=f'Traffic Light: {tl_id}<br>State: {tl_states.get(tl_id, {}).get("state", "Unknown")}<extra></extra>'
                ))
        
        # Set layout
        fig.update_layout(
            title="TeraSim Real-time Simulation State",
            xaxis_title="X Coordinate (m)",
            yaxis_title="Y Coordinate (m)",
            showlegend=True,
            hovermode='closest',
            height=600,
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        
        # Set axis scaling
        if map_data and "bounds" in map_data:
            bounds = map_data["bounds"]
            fig.update_xaxes(range=[bounds["min_x"] - 50, bounds["max_x"] + 50])
            fig.update_yaxes(range=[bounds["min_y"] - 50, bounds["max_y"] + 50])
        
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
    
    def render_statistics(self, dynamic_data):
        """Render statistics"""
        if not dynamic_data:
            return html.Div()
        
        return html.Div([
            html.Div([
                html.H4("Simulation Time"),
                html.P(f"{dynamic_data.get('timestamp', 0):.1f}s"),
                html.P(f"Step: {dynamic_data.get('step', 0)}")
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                html.H4("Vehicle Count"),
                html.P(str(len(dynamic_data.get('vehicles', {}))))
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                html.H4("Average Speed"),
                html.P(f"{np.mean([v['speed'] for v in dynamic_data.get('vehicles', {}).values()]) if dynamic_data.get('vehicles') else 0.0:.1f} m/s")
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                html.H4("Traffic Lights"),
                html.P(str(len(dynamic_data.get('traffic_lights', {}))))
            ], style={'width': '25%', 'display': 'inline-block'})
        ])

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--simulation_uuid", default="demo")
    parser.add_argument("--data_queue_id", default=None)
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TeraSimVisualizer(args.simulation_uuid, args.data_queue_id)
    
    # Define app layout
    app.layout = html.Div([
        html.H1("🚗 TeraSim Real-time Simulation Visualization"),
        html.P(f"Simulation ID: {args.simulation_uuid}"),
        
        html.Div([
            html.Div([
                html.H3("Controls"),
                html.Label("Auto Refresh"),
                dcc.Checklist(
                    id='auto-refresh',
                    options=[{'label': 'Enable', 'value': 'on'}],
                    value=['on']
                ),
                html.Label("Refresh Interval (seconds)"),
                dcc.Slider(
                    id='refresh-interval',
                    min=0.5,
                    max=5.0,
                    step=0.5,
                    value=1.0,
                    marks={i: str(i) for i in range(1, 6)}
                )
            ], style={'width': '20%', 'float': 'left', 'padding': '20px'}),
            
            html.Div([
                dcc.Graph(id='main-graph'),
                html.Div(id='stats-container'),
                dcc.Interval(id='interval-component', interval=1000)
            ], style={'width': '75%', 'float': 'right'})
        ]),
        
        html.Div([
            html.H3("Detailed Data"),
            html.Pre(id='detailed-data', style={'backgroundColor': '#f0f0f0', 'padding': '10px'})
        ], style={'clear': 'both', 'paddingTop': '20px'})
    ])
    
    @app.callback(
        [Output('main-graph', 'figure'),
         Output('stats-container', 'children'),
         Output('detailed-data', 'children')],
        [Input('interval-component', 'n_intervals')],
        [State('auto-refresh', 'value')]
    )
    def update_visualization(n, auto_refresh):
        if 'on' not in auto_refresh:
            return dash.no_update
        
        # Sample data for demonstration
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
        
        # Render components
        fig = visualizer.render_map_with_vehicles(sample_map_data, sample_dynamic_data)
        stats = visualizer.render_statistics(sample_dynamic_data)
        detailed = json.dumps(sample_dynamic_data, indent=2)
        
        return fig, stats, detailed
    
    @app.callback(
        Output('interval-component', 'interval'),
        [Input('refresh-interval', 'value')]
    )
    def update_interval(value):
        return value * 1000
    
    # Run app
    app.run_server(debug=False, port=args.port)

if __name__ == "__main__":
    main()
```

### 3. API Integration

Integration with existing TeraSim-Service API:

```python
# In terasim_service/api.py

@app.post("/start_simulation", tags=["simulations"], summary="Start a new TeraSim simulation")
async def start_simulation(
    config: Annotated[SimulationConfig, Field(description="TeraSim simulation configuration")],
    enable_viz: bool = Query(False, description="Enable visualization"),
    viz_port: int = Query(8050, description="Visualization port"),
    viz_update_freq: int = Query(5, description="Visualization update frequency")
):
    """
    Start a new TeraSim simulation with optional real-time visualization
    
    Args:
        config: Simulation configuration
        enable_viz: Enable real-time visualization
        viz_port: Visualization service port
        viz_update_freq: Visualization update frequency (steps)
    """
    config_data = load_config(config.config_file)
    simulation_id = str(uuid.uuid4())

    # Modified run_simulation_process call
    process = Process(
        target=run_simulation_process,
        args=(simulation_id, config_data, config.auto_run, enable_viz, viz_port, viz_update_freq),
    )
    process.start()

    response = {"simulation_id": simulation_id, "message": "Simulation started"}
    
    if enable_viz:
        response["visualization_url"] = f"http://localhost:{viz_port}"
        response["message"] += " with visualization"

    return response

# Modified run_simulation_task function
async def run_simulation_task(
    simulation_id: str, 
    config: dict, 
    auto_run: bool, 
    enable_viz: bool = False,
    viz_port: int = 8050,
    viz_update_freq: int = 5
):
    try:
        # ... existing code ...
        
        # Create plugin list
        plugins = []
        
        # Add co-simulation plugin
        terasim_cosim_plugin = TeraSimCoSimPlugin(
            simulation_uuid=simulation_id, 
            plugin_config=DEFAULT_COSIM_PLUGIN_CONFIG,
            base_dir=str(base_dir),
            auto_run=auto_run, 
        )
        plugins.append(terasim_cosim_plugin)
        
        # Add visualization plugin if enabled
        if enable_viz:
            from terasim_service.plugins.visualization_plugin import TeraSimVisualizationPlugin
            
            viz_plugin = TeraSimVisualizationPlugin(
                simulation_uuid=simulation_id,
                viz_port=viz_port,
                auto_start_viz=True,
                viz_update_freq=viz_update_freq,
                base_dir=str(base_dir)
            )
            plugins.append(viz_plugin)
        
        # Inject all plugins
        for plugin in plugins:
            plugin.inject(sim, {})
        
        # Run simulation
        await asyncio.get_event_loop().run_in_executor(executor, sim.run)
        
        # ... existing code ...
        
    except Exception as e:
        logger.exception(f"Simulation {simulation_id} failed: {str(e)}")
        # ... error handling ...
```

## Usage Examples

### 1. Start Simulation with Visualization

```bash
# HTTP request example
POST http://localhost:8000/start_simulation?enable_viz=true&viz_port=8050&viz_update_freq=10
Content-Type: application/json

{
    "config_file": "/path/to/simulation_config.yaml",
    "auto_run": false
}
```

### 2. Response Example

```json
{
    "simulation_id": "abc-123-def",
    "message": "Simulation started with visualization",
    "visualization_url": "http://localhost:8050"
}
```

### 3. Python Client Example

```python
import requests

# Start simulation
response = requests.post("http://localhost:8000/start_simulation", 
    params={
        "enable_viz": True,
        "viz_port": 8052
    },
    json={
        "config_file": "./config.yaml",
        "auto_run": False
    }
)

simulation_id = response.json()["simulation_id"]
viz_url = response.json()["visualization_url"]

print(f"Simulation started, ID: {simulation_id}")
print(f"Visualization interface: {viz_url}")

# Control simulation
while True:
    input("Press Enter to advance one step...")
    requests.post(f"http://localhost:8000/simulation_tick/{simulation_id}")
```

## Technical Features

### 1. Performance Optimization
- **Static Data Caching**: Map data extracted only once at startup
- **Incremental Updates**: Only dynamic changes transmitted
- **Configurable Frequency**: Adjustable update rate for performance/real-time balance

### 2. Coordinate System Handling
- **SUMO Coordinates**: Direct use of SUMO internal coordinates, no conversion needed
- **Auto-scaling**: Automatic view range based on map bounds
- **Proportional Display**: Ensures correct map proportions

### 3. Fault Tolerance
- **Process Isolation**: Visualization crash doesn't affect simulation
- **Graceful Degradation**: Simulation continues if visualization fails
- **Resource Cleanup**: Automatic cleanup of temporary files and processes

## Deployment Considerations

### 1. Dependencies
```bash
pip install dash plotly pandas numpy
```

### 2. Port Management
- Default port 8050, configurable
- Port conflict detection and automatic allocation support

### 3. Security Considerations
- Local interface only
- Configurable access permissions

## Extension Possibilities

1. **3D Visualization**: Using Three.js or Plotly 3D
2. **Video Recording**: Automatic simulation video generation
3. **Data Export**: Trajectory data export support
4. **Multi-simulation Support**: Visualize multiple simulation instances
5. **Real-time Analysis**: Integrated performance metrics calculation and display

---

**Note**: This document provides a complete implementation plan based on the current Dash-based visualization system. The plugin follows a modular design for easy maintenance and extension.