import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
import importlib
from loguru import logger
import uuid
from terasim.simulator import Simulator
from terasim.logger.infoextractor import InfoExtractor
from terasim_nde_nade.vehicle.nde_vehicle_factory import NDEVehicleFactory
from terasim_control_plugin import TeraSimControlPlugin
from multiprocessing import Process
import redis
import sys
import json

description = """
TeraSim Control Service API allows you to manage and control TeraSim simulations.
You can start new simulations, check their status, and control ongoing simulations.
"""

tags_metadata = [
    {
        "name": "simulations",
        "description": "Operations with simulations. You can start, stop, and control simulations.",
    },
]

app = FastAPI(
    title="TeraSim Control Service",
    description=description,
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Use a thread pool to limit the number of concurrent simulations
executor = ThreadPoolExecutor(
    max_workers=1
)  # Adjust max_workers based on your system's capacity

# Store running simulation tasks
running_simulations = {}


class SimulationConfig(BaseModel):
    config_file: str = Field(
        ..., description="Path to the simulation configuration file"
    )
    auto_run: bool = Field(
        False,
        description="Whether to automatically run the simulation or wait for manual control",
    )


class SimulationStatus(BaseModel):
    id: str = Field(..., description="Unique identifier for the simulation")
    status: str = Field(..., description="Current status of the simulation")
    progress: float = Field(
        0.0, description="Progress of the simulation as a percentage"
    )


class SimulationCommand(BaseModel):
    command: str = Field(
        ...,
        description="Control command for the simulation (e.g., 'pause', 'resume', 'stop')",
    )


class VehicleState(BaseModel):
    position: tuple[float, float] | None = Field(
        None, description="Vehicle position (x,y)"
    )
    speed: float | None = Field(None, description="Vehicle speed")
    angle: float | None = Field(None, description="Vehicle angle")


class VehicleCommand(BaseModel):
    vehicle_id: str = Field(..., description="ID of the target vehicle")
    type: str = Field(..., description="Command type (e.g., 'set_state')")
    position: tuple[float, float] | None = None
    speed: float | None = None
    angle: float | None = None


class VehicleCommandBatch(BaseModel):
    commands: list[VehicleCommand] = Field(
        ..., description="List of vehicle commands to execute"
    )


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def create_environment(config, base_dir):
    env_module = importlib.import_module(config["environment"]["module"])
    env_class = getattr(env_module, config["environment"]["class"])

    return env_class(
        vehicle_factory=NDEVehicleFactory(),
        info_extractor=InfoExtractor,
        log_flag=config["environment"]["parameters"]["log_flag"],
        log_dir=base_dir,
        warmup_time_lb=config["environment"]["parameters"]["warmup_time_lb"],
        warmup_time_ub=config["environment"]["parameters"]["warmup_time_ub"],
        run_time=config["environment"]["parameters"]["run_time"],
    )


def create_simulator(config, base_dir):
    return Simulator(
        sumo_net_file_path=Path(config["file_paths"]["sumo_net_file"]),
        sumo_config_file_path=Path(config["file_paths"]["sumo_config_file"]),
        num_tries=config["simulator"]["parameters"]["num_tries"],
        gui_flag=config["simulator"]["parameters"]["gui_flag"],
        realtime_flag=config["simulator"]["parameters"].get("realtime_flag", False),
        output_path=base_dir,
        sumo_output_file_types=config["simulator"]["parameters"][
            "sumo_output_file_types"
        ],
    )


async def run_simulation_task(simulation_id: str, config: dict, auto_run: bool):
    try:
        base_dir = (
            Path(config["output"]["dir"])
            / config["output"]["name"]
            / "raw_data"
            / config["output"]["nth"]
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        env = create_environment(config, base_dir)
        sim = create_simulator(config, base_dir)
        sim.bind_env(env)

        # Create and inject TeraSimControlPlugin
        control_plugin = TeraSimControlPlugin(
            simulation_uuid=simulation_id, auto_run=auto_run
        )
        control_plugin.inject(sim, {})

        # Run the simulation
        await asyncio.get_event_loop().run_in_executor(executor, sim.run)

        running_simulations[simulation_id].status = "completed"
    except Exception as e:
        logger.exception(f"Simulation {simulation_id} failed: {str(e)}")
        running_simulations[simulation_id].status = "failed"
    finally:
        # Remove the simulation from running_simulations
        del running_simulations[simulation_id]

        # Force garbage collection
        import gc

        gc.collect()


def run_simulation_process(simulation_id: str, config: dict, auto_run: bool):
    # This function will run in a separate process
    asyncio.run(run_simulation_task(simulation_id, config, auto_run))


@app.post("/start_simulation", tags=["simulations"], summary="Start a new simulation")
async def start_simulation(config: SimulationConfig):
    """
    Start a new simulation with the given configuration.

    Parameters:
    - config: SimulationConfig object containing the simulation configuration

    Returns:
    - dict: A dictionary containing the simulation_id and a status message

    Example:
    ```
    {
        "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": "Simulation started"
    }
    ```
    """
    config_data = load_config(config.config_file)
    simulation_id = str(uuid.uuid4())
    running_simulations[simulation_id] = SimulationStatus(
        id=simulation_id, status="running"
    )

    # Start the simulation in a new process
    process = Process(
        target=run_simulation_process,
        args=(simulation_id, config_data, config.auto_run),
    )
    process.start()

    return {"simulation_id": simulation_id, "message": "Simulation started"}


def get_simulation_id(simulation_id: str):
    if simulation_id not in running_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation_id


def check_simulation_running(simulation_id: str, redis_client: redis.Redis) -> bool:
    """Check if a simulation is still running in Redis

    Returns:
        bool: True if simulation is running, False if stopped or doesn't exist
    """
    status = redis_client.get(f"sim:{simulation_id}:status")
    return status is not None and status.decode("utf-8") != "finished"


@app.get(
    "/simulation_status/{simulation_id}",
    tags=["simulations"],
    summary="Get simulation status",
    responses={
        200: {"description": "Successful response", "model": SimulationStatus},
        404: {"description": "Simulation not found"},
    },
)
async def get_simulation_status(simulation_id: str = Depends(get_simulation_id)):
    """
    Get the current status of a simulation.

    Parameters:
    - simulation_id: Unique identifier of the simulation

    Returns:
    - SimulationStatus: Object containing the current status of the simulation
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )
        return running_simulations[simulation_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simulation_control/{simulation_id}",
    tags=["simulations"],
    summary="Control a simulation",
    responses={
        200: {"description": "Command sent successfully"},
        404: {"description": "Simulation not found"},
        500: {"description": "Failed to send command"},
    },
)
async def control_simulation(
    simulation_id: str = Depends(get_simulation_id),
    command: SimulationCommand = Body(...),
):
    """
    Send a control command to an ongoing simulation.

    Parameters:
    - simulation_id: Unique identifier of the simulation
    - command: SimulationCommand object containing the control command

    Returns:
    - dict: A dictionary containing a message about the sent command
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        if command.command == "resume":
            redis_client.delete(f"sim:{simulation_id}:paused")
        elif command.command == "pause":
            redis_client.set(f"sim:{simulation_id}:paused", "1")
        elif command.command == "stop":
            redis_client.set(f"sim:{simulation_id}:control", command.command)

        return {
            "message": f"Command '{command.command}' sent to simulation {simulation_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send command: {str(e)}")


@app.post(
    "/simulation_tick/{simulation_id}",
    tags=["simulations"],
    summary="Advance simulation by one step",
    responses={
        200: {"description": "Tick command sent successfully"},
        404: {"description": "Simulation not found"},
        500: {"description": "Failed to send tick command"},
    },
)
async def tick_simulation(simulation_id: str = Depends(get_simulation_id)):
    """
    Advance the simulation by one step. This only works when auto_run is false.

    Parameters:
    - simulation_id: Unique identifier of the simulation

    Returns:
    - dict: A dictionary containing a message about the sent tick command
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        redis_client.set(f"sim:{simulation_id}:control", "tick")
        return {"message": f"Tick command sent to simulation {simulation_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send tick command: {str(e)}"
        )


@app.get(
    "/simulation_results/{simulation_id}",
    tags=["simulations"],
    summary="Get simulation results",
    responses={
        200: {"description": "Simulation results retrieved successfully"},
        404: {"description": "Simulation not found"},
    },
)
async def get_simulation_results(simulation_id: str = Depends(get_simulation_id)):
    """
    Retrieve the results of a completed simulation.

    Parameters:
    - simulation_id: Unique identifier of the simulation

    Returns:
    - dict: A dictionary containing the simulation status and results (if available)
    """
    status = running_simulations[simulation_id].status
    if status == "running":
        return {
            "simulation_id": simulation_id,
            "status": status,
            "message": "Simulation still running",
        }
    elif status == "failed":
        return {
            "simulation_id": simulation_id,
            "status": status,
            "message": "Simulation failed",
        }
    elif status == "completed":
        # Implement result retrieval logic here
        # This might involve reading files from the output directory or querying a database
        results = (
            "Simulation results here"  # Replace with actual result retrieval logic
        )
        return {"simulation_id": simulation_id, "status": status, "results": results}


@app.get(
    "/simulation/{simulation_id}/vehicles",
    tags=["vehicles"],
    summary="Get all vehicle states",
)
async def get_vehicle_states(simulation_id: str = Depends(get_simulation_id)):
    """
    Get the current state of all vehicles in the simulation.
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        state = redis_client.hget(f"sim:{simulation_id}:state", "data")
        if not state:
            raise HTTPException(status_code=404, detail="Simulation state not found")

        return json.loads(state.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simulation/{simulation_id}/vehicle_command",
    tags=["vehicles"],
    summary="Send command to control a vehicle",
)
async def control_vehicle(
    simulation_id: str = Depends(get_simulation_id), command: VehicleCommand = Body(...)
):
    """
    Send a command to control a specific vehicle in the simulation.
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        redis_client.rpush(
            f"sim:{simulation_id}:vehicle_commands", json.dumps(command.dict())
        )
        return {"message": f"Vehicle command sent for vehicle {command.vehicle_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simulation/{simulation_id}/vehicle_commands_batch",
    tags=["vehicles"],
    summary="Send multiple commands to control vehicles",
)
async def control_vehicles_batch(
    simulation_id: str = Depends(get_simulation_id),
    command_batch: VehicleCommandBatch = Body(...),
):
    """
    Send multiple commands to control different vehicles in the simulation.
    Commands will be executed in the order they are provided.
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        # Add all commands to the queue
        for command in command_batch.commands:
            redis_client.rpush(
                f"sim:{simulation_id}:vehicle_commands", json.dumps(command.dict())
            )
        return {
            "message": f"Batch of {len(command_batch.commands)} vehicle commands sent successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add this function to check Redis connection
def check_redis_connection():
    try:
        redis_client = redis.Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        logger.info("Successfully connected to Redis")
    except redis.ConnectionError:
        logger.error("Failed to connect to Redis. Exiting...")
        sys.exit(1)


if __name__ == "__main__":
    import uvicorn

    # Check Redis connection before starting the service
    check_redis_connection()

    uvicorn.run(app, host="0.0.0.0", port=8000)
    # start simulation here
    # config_file = (
    #     "/home/haoweis/TeraSim_Development/TeraSim-Service/simulation_config.yaml"
    # )
    # auto_run = True
    # simulation_config = SimulationConfig(config_file=config_file, auto_run=auto_run)

    # asyncio.run(start_simulation(simulation_config))
