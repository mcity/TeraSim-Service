# TeraSim Service

A HTTP service for TeraSim simulation control and monitoring, built with FastAPI.

## Overview

TeraSim Service provides a comprehensive HTTP API for managing simulation instances, enabling remote control and real-time monitoring capabilities.

## Core Functionalities

- Start new simulation instances
- Check simulation status
- Control simulations (pause, resume, stop)
- Manually advance simulation steps (tick)
- Retrieve simulation results

## API Endpoints

```
POST /start_simulation              # Start a new simulation
GET  /simulation_status/{sim_id}    # Check simulation status
POST /simulation_control/{sim_id}   # Control simulation
POST /simulation_tick/{sim_id}      # Advance simulation step
GET  /simulation_results/{sim_id}   # Retrieve simulation results
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

4. Install service dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the service:
   ```bash
   python terasim_service.py
   ```

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2024 TeraSim Service Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the GitHub repository or contact [your contact information].