"""
TeraSim Real-time Visualization Application (Dash Version)

This Dash application provides real-time visualization of TeraSim simulations.
It reads simulation state from Redis and displays vehicles, traffic lights, and road network.
Uses Dash's callback system for efficient updates without page refresh.
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import json
import argparse
import redis
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--simulation_uuid", required=True)
parser.add_argument("--redis_host", default="localhost")
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--port", type=int, default=8501)
parser.add_argument("--update_interval", type=float, default=0.5, help="Update interval in seconds")
args = parser.parse_args()

# Connect to Redis
redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

# Initialize Dash app
app = dash.Dash(__name__, update_title=None)
app.title = "TeraSim Real-time Visualization"

# Define CSS styles
styles = {
    'container': {
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif'
    },
    'header': {
        'backgroundColor': '#f0f0f0',
        'padding': '10px',
        'marginBottom': '20px',
        'borderRadius': '5px'
    },
    'title': {
        'fontSize': '24px',
        'fontWeight': 'bold',
        'marginBottom': '10px'
    },
    'info': {
        'fontSize': '14px',
        'color': '#666'
    },
    'controlPanel': {
        'backgroundColor': '#f8f8f8',
        'padding': '15px',
        'borderRadius': '5px',
        'marginBottom': '20px'
    },
    'metric': {
        'backgroundColor': 'white',
        'padding': '10px',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '10px'
    },
    'metricTitle': {
        'fontSize': '12px',
        'color': '#666',
        'marginBottom': '5px'
    },
    'metricValue': {
        'fontSize': '20px',
        'fontWeight': 'bold'
    }
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚗 TeraSim Real-time Visualization", style=styles['title']),
        html.Div(f"Simulation ID: {args.simulation_uuid}", style=styles['info'])
    ], style=styles['header']),
    
    # Main content
    html.Div([
        # Left column - Map
        html.Div([
            dcc.Graph(
                id='live-map',
                style={'height': '700px'},
                config={'displayModeBar': True, 'displaylogo': False}
            )
        ], style={'width': '75%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right column - Statistics and Controls
        html.Div([
            # Control Panel
            html.Div([
                html.H3("Control Panel", style={'marginBottom': '15px'}),
                html.Label("Refresh Interval (seconds):", style={'marginRight': '10px'}),
                dcc.Slider(
                    id='interval-slider',
                    min=0.1,
                    max=2.0,
                    step=0.1,
                    value=args.update_interval,  # Use the provided interval
                    marks={i/10: str(i/10) for i in range(1, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(style={'marginBottom': '20px'}),
                
                # Checkboxes
                dcc.Checklist(
                    id='display-options',
                    options=[
                        {'label': ' Show Vehicle Labels', 'value': 'labels'},
                        {'label': ' Show Traffic Lights', 'value': 'traffic_lights'},
                        {'label': ' Show Construction Zones', 'value': 'construction'}
                    ],
                    value=['labels', 'traffic_lights', 'construction'],
                    style={'marginBottom': '10px'}
                )
            ], style=styles['controlPanel']),
            
            # Statistics
            html.Div([
                html.H3("📊 Statistics", style={'marginBottom': '15px'}),
                html.Div(id='stats-container')
            ], style={'marginTop': '20px'}),
            
            # Status
            html.Div([
                html.H3("🔄 Status", style={'marginBottom': '15px'}),
                html.Div(id='status-container')
            ], style={'marginTop': '20px'})
            
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    
    # Interval component for updates
    dcc.Interval(
        id='interval-component',
        interval=int(args.update_interval * 1000),  # Convert to milliseconds
        n_intervals=0
    ),
    
    # Store components for data
    dcc.Store(id='map-data-store'),
    dcc.Store(id='previous-state-store')
    
], style=styles['container'])


def get_simulation_state(simulation_uuid):
    """Get current simulation state from Redis."""
    try:
        state_str = redis_client.get(f"simulation:{simulation_uuid}:state")
        if state_str:
            return json.loads(state_str)
    except Exception as e:
        logger.error(f"Error getting simulation state: {e}")
    return None


def get_map_data(simulation_uuid):
    """Get static map data from Redis."""
    try:
        map_data_str = redis_client.get(f"simulation:{simulation_uuid}:map_data")
        if map_data_str:
            return json.loads(map_data_str)
    except Exception as e:
        logger.error(f"Error getting map data: {e}")
    return None


def create_map_traces(map_data, sim_state, display_options):
    """Create all map traces for the visualization."""
    traces = []
    
    # Draw lanes
    if map_data and "lanes" in map_data:
        for lane in map_data["lanes"]:
            x_coords = [pt[0] for pt in lane["shape"]]
            y_coords = [pt[1] for pt in lane["shape"]]
            
            traces.append(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='lightgray', width=lane["width"]*2),
                name=f'Lane {lane["id"]}',
                showlegend=False,
                hovertemplate=f'Lane: {lane["id"]}<br>Speed limit: {lane["speed_limit"]:.1f} m/s<extra></extra>'
            ))
    
    # Draw junctions
    if map_data and "junctions" in map_data:
        for junction in map_data["junctions"]:
            if junction["shape"]:
                x_coords = [pt[0] for pt in junction["shape"]] + [junction["shape"][0][0]]
                y_coords = [pt[1] for pt in junction["shape"]] + [junction["shape"][0][1]]
                
                traces.append(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    fill='toself',
                    fillcolor='rgba(135, 206, 235, 0.2)',
                    line=dict(color='steelblue', width=1),
                    showlegend=False,
                    hovertemplate=f'Junction: {junction["id"]}<extra></extra>'
                ))
    
    # Draw vehicles
    if sim_state and "agent_details" in sim_state:
        vehicles = sim_state["agent_details"].get("vehicle", {})
        show_labels = 'labels' in display_options
        
        if vehicles:
            # Separate vehicles by type
            vehicle_types = {
                'AV': {'x': [], 'y': [], 'text': [], 'hover': [], 'color': 'red'},
                'BV': {'x': [], 'y': [], 'text': [], 'hover': [], 'color': 'blue'},
                'Other': {'x': [], 'y': [], 'text': [], 'hover': [], 'color': 'green'}
            }
            
            for vid, vdata in vehicles.items():
                x, y = vdata["x"], vdata["y"]
                text = vid if show_labels else ""
                hover = f'{vid}<br>Speed: {vdata["speed"]:.1f} m/s<br>Type: {vdata["type"]}'
                
                if "AV" in vid:
                    vtype = 'AV'
                elif "BV" in vid:
                    vtype = 'BV'
                else:
                    vtype = 'Other'
                
                vehicle_types[vtype]['x'].append(x)
                vehicle_types[vtype]['y'].append(y)
                vehicle_types[vtype]['text'].append(text)
                vehicle_types[vtype]['hover'].append(hover)
            
            # Add vehicle traces
            for vtype, vdata in vehicle_types.items():
                if vdata['x']:
                    traces.append(go.Scatter(
                        x=vdata['x'],
                        y=vdata['y'],
                        mode='markers+text',
                        marker=dict(size=10, color=vdata['color'], symbol='triangle-up'),
                        text=vdata['text'],
                        textposition="top center",
                        name=vtype,
                        hovertext=vdata['hover'],
                        hovertemplate='%{hovertext}<extra></extra>'
                    ))
        
        # Draw VRUs (pedestrians)
        vrus = sim_state["agent_details"].get("vru", {})
        if vrus:
            vru_x = [v["x"] for v in vrus.values()]
            vru_y = [v["y"] for v in vrus.values()]
            vru_text = [vid if show_labels else "" for vid in vrus.keys()]
            
            traces.append(go.Scatter(
                x=vru_x, y=vru_y,
                mode='markers+text',
                marker=dict(size=8, color='orange', symbol='circle'),
                text=vru_text,
                textposition="top center",
                name='VRU',
                hovertemplate='VRU: %{text}<extra></extra>'
            ))
    
    # Draw traffic lights
    if 'traffic_lights' in display_options and map_data and "traffic_lights" in map_data:
        tl_states = {}
        if sim_state and "traffic_light_details" in sim_state:
            tl_states = sim_state["traffic_light_details"]
        
        for tl in map_data["traffic_lights"]:
            tl_id = tl["id"]
            x, y = tl["position"]
            
            # Get current state
            color = 'gray'
            state = 'unknown'
            if tl_id in tl_states:
                state_str = tl_states[tl_id].get("tls", "")
                if 'r' in state_str.lower():
                    color = 'red'
                    state = 'red'
                elif 'y' in state_str.lower():
                    color = 'yellow'
                    state = 'yellow'
                elif 'g' in state_str.lower():
                    color = 'green'
                    state = 'green'
            
            traces.append(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='black')),
                showlegend=False,
                hovertemplate=f'Traffic Light: {tl_id}<br>State: {state}<extra></extra>'
            ))
    
    # Draw construction zones
    if 'construction' in display_options and sim_state and "construction_zone_details" in sim_state:
        construction_zones = sim_state["construction_zone_details"]
        if construction_zones:
            for lane_id, shape in construction_zones.items():
                x_coords = [pt[0] for pt in shape]
                y_coords = [pt[1] for pt in shape]
                
                traces.append(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='lines',
                    line=dict(color='orange', width=3, dash='dash'),
                    showlegend=False,
                    hovertemplate=f'Construction Zone<br>Lane: {lane_id}<extra></extra>'
                ))
    
    return traces


# Callback to load initial map data
@app.callback(
    Output('map-data-store', 'data'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=False
)
def load_map_data(n):
    """Load map data once and store it."""
    if n == 0:  # Only load on first interval
        return get_map_data(args.simulation_uuid)
    return dash.no_update


# Callback to update interval
@app.callback(
    Output('interval-component', 'interval'),
    Input('interval-slider', 'value')
)
def update_interval(value):
    """Update the refresh interval."""
    return value * 1000  # Convert to milliseconds


# Main update callback
@app.callback(
    [Output('live-map', 'figure'),
     Output('stats-container', 'children'),
     Output('status-container', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('map-data-store', 'data'),
     State('display-options', 'value')]
)
def update_visualization(n, map_data, display_options):
    """Update the visualization with current simulation state."""
    
    # Debug logging
    logger.info(f"Update callback triggered: n_intervals={n}")
    
    # Get current simulation state
    sim_state = get_simulation_state(args.simulation_uuid)
    
    if sim_state:
        logger.info(f"Got simulation state, time={sim_state.get('simulation_time', -1)}")
    else:
        logger.warning("No simulation state found!")
    
    # Create figure
    fig = go.Figure()
    
    if sim_state and map_data:
        # Add all traces
        traces = create_map_traces(map_data, sim_state, display_options)
        for trace in traces:
            fig.add_trace(trace)
        
        # Update layout
        fig.update_layout(
            title="TeraSim Simulation Map",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            hovermode='closest',
            height=700,
            showlegend=True,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            uirevision='constant'  # Keep zoom/pan state
        )
        
        # Set axis properties
        if "bounds" in map_data:
            bounds = map_data["bounds"]
            margin = 50
            fig.update_xaxes(range=[bounds["min_x"]-margin, bounds["max_x"]+margin])
            fig.update_yaxes(range=[bounds["min_y"]-margin, bounds["max_y"]+margin])
        
        # Equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        # Create statistics
        vehicles = sim_state.get("agent_details", {}).get("vehicle", {})
        vrus = sim_state.get("agent_details", {}).get("vru", {})
        
        stats_children = [
            html.Div([
                html.Div("Simulation Time", style=styles['metricTitle']),
                html.Div(f"{sim_state.get('simulation_time', 0):.1f} s", style=styles['metricValue'])
            ], style=styles['metric']),
            html.Div([
                html.Div("Vehicles", style=styles['metricTitle']),
                html.Div(str(len(vehicles)), style=styles['metricValue'])
            ], style=styles['metric']),
            html.Div([
                html.Div("VRUs", style=styles['metricTitle']),
                html.Div(str(len(vrus)), style=styles['metricValue'])
            ], style=styles['metric'])
        ]
        
        if vehicles:
            avg_speed = np.mean([v["speed"] for v in vehicles.values()])
            stats_children.append(
                html.Div([
                    html.Div("Avg Speed", style=styles['metricTitle']),
                    html.Div(f"{avg_speed:.1f} m/s", style=styles['metricValue'])
                ], style=styles['metric'])
            )
        
        # Create status
        status = redis_client.get(f"simulation:{args.simulation_uuid}:status") or "Unknown"
        status_children = [
            html.Div([
                html.Div(f"✅ Status: {status}", style={'color': 'green', 'fontWeight': 'bold'}),
                html.Div(f"Last update: {datetime.now().strftime('%H:%M:%S')}", 
                        style={'color': '#666', 'fontSize': '12px', 'marginTop': '5px'})
            ])
        ]
        
    else:
        # No data available
        fig.add_annotation(
            text="Waiting for simulation data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        stats_children = [html.Div("No data available", style={'color': 'gray'})]
        status_children = [html.Div("Waiting for simulation...", style={'color': 'orange'})]
    
    return fig, stats_children, status_children


if __name__ == '__main__':
    logger.info(f"Starting Dash visualization app on port {args.port}")
    logger.info(f"Simulation UUID: {args.simulation_uuid}")
    logger.info(f"Redis connection: {args.redis_host}:{args.redis_port}")
    
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=False
    )