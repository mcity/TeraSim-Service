# TeraSim Visualization Usage Guide

## Overview

TeraSim visualization functionality is integrated into the CoSim plugin, providing real-time simulation state display including:
- 🗺️ Map layout (lanes, junctions)
- 🚗 Vehicle positions and states (AV, BV, other vehicles)
- 🚶 Pedestrian (VRU) positions
- 🚦 Traffic light states
- 🚧 Construction zones

## Quick Start

### 1. Start Simulation with Visualization

```python
import requests

# Start simulation with visualization enabled
response = requests.post("http://localhost:8000/start_simulation", 
    json={
        "config_file": "./config.yaml",
        "auto_run": False  # or True
    }, 
    params={
        "enable_viz": True,      # Enable visualization
        "viz_port": 8050,        # Dash port
        "viz_update_freq": 5     # Update every 5 steps
    }
)

result = response.json()
print(f"Simulation ID: {result['simulation_id']}")
print(f"Visualization URL: {result['visualization_url']}")
```

### 2. Access Visualization Interface

Open the returned URL in your browser (default: http://localhost:8050)

## API Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_viz` | bool | False | Enable visualization |
| `viz_port` | int | 8050 | Dash service port |
| `viz_update_freq` | int | 5 | Visualization update frequency (simulation steps) |

## Visualization Interface Features

### Control Panel (Sidebar)
- **Auto Refresh**: Toggle automatic refresh
- **Refresh Interval**: Set refresh interval (0.1-2 seconds)
- **Show Vehicle Labels**: Display vehicle ID labels
- **Show Traffic Lights**: Display traffic lights
- **Show Construction Zones**: Display construction zones

### Main Interface
- **Map View**: Real-time simulation scene display
- **Statistics**: Simulation time, vehicle count, average speed, etc.
- **Status Info**: Current simulation status and last update time

## Vehicle Color Legend
- 🔴 Red Triangle: AV (Autonomous Vehicle)
- 🔵 Blue Triangle: BV (Background Vehicle)
- 🟢 Green Triangle: Other vehicles
- 🟠 Orange Circle: VRU (Vulnerable Road User - pedestrian/cyclist)

## Usage Examples

### Example 1: Manual Control Mode
```bash
# Run test script
python test_visualization.py
# Select 1 (Manual control)
```

### Example 2: Auto-run Mode
```bash
# Run test script
python test_visualization.py
# Select 2 (Auto-run mode)
```

### Example 3: Using Different Port
```python
# If port 8050 is occupied, use another port
response = requests.post("http://localhost:8000/start_simulation", 
    json={"config_file": "./config.yaml", "auto_run": True}, 
    params={"enable_viz": True, "viz_port": 8052}
)
```

## Technical Details

### Data Flow
1. CoSim plugin extracts map data on startup (if visualization enabled)
2. CoSim plugin collects simulation data each step and writes to Redis
3. Dash application reads data from Redis and renders in real-time

### Architecture Advantages
- **Integrated Design**: Visualization as optional feature of CoSim plugin
- **Backward Compatible**: Behaves identically when disabled
- **Resource Optimized**: Reuses existing data collection mechanism

### Performance Optimization
- Map data extracted only once at startup
- Configurable update frequency to balance performance and real-time updates
- Uses Plotly for optimized rendering performance

## Troubleshooting

### Dash Cannot Start
- Check if port is occupied
- Ensure dependencies installed: `pip install dash plotly`

### Cannot See Vehicles
- Confirm simulation is running
- Check Redis connection status
- View browser console for error messages

### Performance Issues
- Increase `viz_update_freq` value to reduce update frequency
- Disable unnecessary display options
- Test with smaller simulation scenarios