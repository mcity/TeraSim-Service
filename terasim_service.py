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
    return running_simulations[simulation_id]


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
        import redis

        redis_client = redis.Redis()
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
        import redis

        redis_client = redis.Redis()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
