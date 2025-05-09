#!/bin/bash
# Start Redis server
redis-server &

# Start the TeraSim server
python examples/main.py &

# Wait for the TeraSim server to start
sleep 30

# Start the Carla server
python examples/carla/main.py --terasim_config /app/examples/simulation_Mcity_carla_config.yaml --map_name=McityMap_Main --control_av