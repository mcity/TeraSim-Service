# TeraSim Service

A HTTP service for TeraSim simulation control and monitoring, built with FastAPI.

## Overview

TeraSim Service provides a comprehensive HTTP API for managing simulation instances, enabling remote control and real-time monitoring capabilities.

## Core Functionalities

- Start new simulation instances
- Check simulation status

   The simulation will have the following statuses and their meanings are shown below:
   | Simulation Status | Meaning                                                                 |
   |-------------------|-------------------------------------------------------------------------|
   | "initializing"         | The simulation has started for traffic flow initialization.             |
   | "wait_for_tick"     | The traffic flow has been initialized and the simulation is waiting for the "tick" command. |
   | "running"         | After receiving the "tick" command, the TeraSim simulation has started to advance for one step and it will take hundreds of milliseconds to finish the calculation. |
   | "ticked"         | The TeraSim simulation has finished the one-step advance and it is waiting for the next "tick" command. |
   | "finished"        | The TeraSim simulation has come to an end.                              |

- Control simulations (pause, resume, stop)
- Manually advance simulation steps (tick)
- Retrieve simulation states
- Send commands to agents

## API Endpoints

```
POST /start_simulation                    # Start a new simulation
GET  /simulation_status/{sim_id}          # Check simulation status
POST /simulation_control/{sim_id}         # Control simulation
POST /simulation_tick/{sim_id}            # Advance simulation step
GET  /simulation/{sim_id}/state           # Retrieve simulation states
POST /simulation/{sim_id}/agent_command   # Send command to agent
```

## Requirements

- Python 3.9+
- Redis server
- TeraSim and its dependencies

## Setup

1. Install TeraSim package:
   ```bash
   pip install terasim
   ```

2. Install Redis:
   For details, check [this link](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/install-redis-on-linux/)

3. Start Redis server:
   ```bash
   sudo systemctl enable redis-server
   sudo systemctl start redis-server
   # Verify Redis is running
   redis-cli ping
   # Should return "PONG"
   ```

4. Install TeraSim-Service package:
   ```bash
   poetry install
   ```

5. Start the service:
   ```bash
   python examples/main.py
   ```

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the GitHub repository.