import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path

import redis
from fastapi import Body, Depends, FastAPI, HTTPException
from loguru import logger


from terasim_service.plugins import TeraSimCoSimPluginBefore, TeraSimCoSimPluginAfter, COSIM_PLUGIN_BEFORE_CONFIG
from terasim_service.utils import (
    check_redis_connection,
    create_environment,
    create_simulator,
    load_config,
    SimulationConfig,
    SimulationCommand,
    SimulationStatus,
    AgentCommand,
    AgentCommandBatch,
)

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



async def run_simulation_task(simulation_id: str, config: dict, auto_run: bool):
    try:
        base_dir = (
            Path(config["output"]["dir"])
            / config["output"]["name"]
            / "raw_data"
            / config["output"]["nth"]
            /simulation_id
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        env = create_environment(config, base_dir)
        sim = create_simulator(config, base_dir)
        sim.bind_env(env)

        # Create and inject TeraSimControlPlugin
        control_plugin_before_config = COSIM_PLUGIN_BEFORE_CONFIG
        control_plugin_before_config["centered_agent_ID"] = "CAV"
        control_plugin_before = TeraSimCoSimPluginBefore(
            simulation_uuid=simulation_id, 
            plugin_config=control_plugin_before_config,
            base_dir=str(base_dir),
            auto_run=auto_run, 
        )
        control_plugin_before.inject(sim, {})
        control_plugin_after = TeraSimCoSimPluginAfter(
            simulation_uuid=simulation_id, 
            base_dir=str(base_dir),
            auto_run=auto_run
        )
        control_plugin_after.inject(sim, {})

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
        id=simulation_id, status="started"
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
    status = redis_client.get(f"simulation:{simulation_id}:status")
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
        running_simulations[simulation_id].status = redis_client.get(f"simulation:{simulation_id}:status").decode("utf-8")
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
            redis_client.delete(f"simulation:{simulation_id}:paused")
        elif command.command == "pause":
            redis_client.set(f"simulation:{simulation_id}:paused", "1")
        elif command.command == "stop":
            redis_client.set(f"simulation:{simulation_id}:control", command.command)

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

        redis_client.set(f"simulation:{simulation_id}:control", "tick")
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
    "/simulation/{simulation_id}/state",
    tags=["simulations"],
    summary="Get all simulation states",
)
async def get_simulation_state(simulation_id: str = Depends(get_simulation_id)):
    """
    Get the current state of the simulation.
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        state = redis_client.get(f"simulation:{simulation_id}:state")
        if not state:
            raise HTTPException(status_code=404, detail="Simulation state not found")

        return json.loads(state.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simulation/{simulation_id}/agent_command",
    tags=["agents"],
    summary="Send command to control an agent",
)
async def control_vehicle(
    simulation_id: str = Depends(get_simulation_id), command: AgentCommand = Body(...)
):
    """
    Send a command to control a specific agent in the simulation.
    """
    try:
        redis_client = redis.Redis()
        if not check_simulation_running(simulation_id, redis_client):
            raise HTTPException(
                status_code=404, detail="Simulation not found or finished"
            )

        redis_client.rpush(
            f"simulation:{simulation_id}:agent_commands", json.dumps(command.dict())
        )
        return {"message": f"Agent command sent for agent {command.agent_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/simulation/{simulation_id}/agent_commands_batch",
    tags=["agents"],
    summary="Send multiple commands to control agents",
)
async def control_agents_batch(
    simulation_id: str = Depends(get_simulation_id),
    command_batch: AgentCommandBatch = Body(...),
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
                f"simulation:{simulation_id}:agent_commands", json.dumps(command.dict())
            )
        return {
            "message": f"Batch of {len(command_batch.commands)} agent commands sent successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Check Redis connection before starting the service
    check_redis_connection()

    uvicorn.run(app, host="0.0.0.0", port=8000)
