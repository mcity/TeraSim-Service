# TeraSim Control Service

TeraSim Control Service is a FastAPI-based web service that allows users to start, control, and monitor TeraSim simulations remotely. It provides a RESTful API for managing simulation instances and interacting with them in real-time.

## Features

- Start new simulation instances
- Check simulation status
- Control simulations (pause, resume, stop)
- Manually advance simulation steps (tick)
- Retrieve simulation results

## Prerequisites

- Python 3.7+
- Redis server
- TeraSim and its dependencies

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/TeraSim-Service.git
   cd TeraSim-Service
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure Redis is installed and running on your system.

## Configuration

1. Update the `simulation_config.yaml` file with your desired simulation parameters.

2. Modify the `terasim_service.py` file if you need to adjust the maximum number of concurrent simulations or other service-specific settings.

## Usage

1. Start the TeraSim Control Service:
   ```
   python terasim_service.py
   ```

2. Use the provided REST API endpoints to interact with the service:

   - Start a new simulation:
     ```
     POST /start_simulation
     ```

   - Check simulation status:
     ```
     GET /simulation_status/{simulation_id}
     ```

   - Control simulation:
     ```
     POST /simulation_control/{simulation_id}
     ```

   - Advance simulation step:
     ```
     POST /simulation_tick/{simulation_id}
     ```

   - Retrieve simulation results:
     ```
     GET /simulation_results/{simulation_id}
     ```

3. You can use the `terasim_request.rest` file as a reference for making API calls using a REST client.

## API Documentation

Once the service is running, you can access the auto-generated API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Support

For support, please open an issue in the GitHub repository or contact [your contact information].