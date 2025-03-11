import json
import logging
from logging.handlers import RotatingFileHandler
import redis
from redis.exceptions import RedisError
import time

from terasim.overlay import traci
from terasim.simulator import Simulator

from .base import BasePlugin, DEFAULT_REDIS_CONFIG

from ..utils import SimulationState, AgentStateSimplified, SUMOSignal, AgentCommand


DEFAULT_COSIM_PLUGIN_CONFIG = {
    "name": "terasim_cosim_plugin",
    "priority": {
        "start": -90,
        "step": -90,
        "stop": -90,
    },
}

COSIM_PLUGIN_BEFORE_CONFIG = {
    "name": "terasim_cosim_plugin_before",
    "priority": {
        "start": -90,
        "step": -90,
        "stop": -90,
    },
}

COSIM_PLUGIN_AFTER_CONFIG = {
    "name": "terasim_cosim_plugin_after",
    "priority": {
        "start": 90,
        "step": 90,
        "stop": 90,
    },
}


class TeraSimCoSimPlugin(BasePlugin):
    def __init__(
        self,
        simulation_uuid: str,
        plugin_config: dict = DEFAULT_COSIM_PLUGIN_CONFIG,
        redis_config: dict = DEFAULT_REDIS_CONFIG,
        key_expiry=3600,
        auto_run=False,
    ):
        """Initialize the Co-Simulation plugin.

        Args:
            simulation_uuid (str): Unique identifier for the simulation instance.
            plugin_config (dict, optional): Configuration for the plugin. Defaults to DEFAULT_COSIM_PLUGIN_CONFIG.
            redis_config (dict, optional): Configuration for the Redis connection. Defaults to DEFAULT_REDIS_CONFIG.
            key_expiry (int, optional): Key expiration time in seconds. Defaults to 3600.
            auto_run (bool, optional): Flag to enable auto-run mode. Defaults to False.
        """
        super().__init__(simulation_uuid, plugin_config, redis_config)
        # Key expiration time in seconds (default: 1 hour)
        self.key_expiry = key_expiry
        self.auto_run = auto_run

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logger for the plugin.

        Returns:
            logging.Logger: Logger instance for the plugin.
        """
        logger = logging.getLogger(f"{self.plugin_name}-{self.simulation_uuid}")
        logger.setLevel(logging.DEBUG)

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            f"{self.plugin_name}_{self.simulation_uuid}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def on_start(self, simulator: Simulator, ctx):
        """Connect to the Redis server and set the simulation status to be 'started'.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        pass

    def on_step(self, simulator: Simulator, ctx):
        """Handle simulation step logic, including handling simulation level commands, handling agent-level command, and retrieving simulation states.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        
        Returns:
            bool: True if the simulation step was successful, False otherwise.
        """
        pass

    def on_stop(self, simulator: Simulator, ctx):
        """Stop the simulation and clean up all simulation related data.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        pass

    def _check_simulation_status(self) -> bool:
        """Check if simulation is still running.

        Returns:
            bool: True if simulation is running, False if stopped or doesn't exist
        """
        status = self.redis_client.get(f"simulation:{self.simulation_uuid}:status")
        if not status or status.decode("utf-8") == "finished":
            self.logger.warning(
                f"Simulation {self.simulation_uuid} is stopped or doesn't exist"
            )
            return False
        return True

    def _get_and_handle_command(self, simulator: Simulator) -> str | None:
        """Get and handle simulation control commands.

        Args:
            simulator (Simulator): The simulator object.

        Returns:
            str | None: The control command to execute, or None if no command is present.
        """
        if not self._check_simulation_status():
            return "stop"
        command = self.redis_client.get(f"simulation:{self.simulation_uuid}:control")
        if command:
            command = command.decode("utf-8")
            self._handle_control_command(command, simulator)
            if command != "stop":
                self.redis_client.delete(f"simulation:{self.simulation_uuid}:control")
        return command

    def _is_simulation_paused(self) -> bool:
        """Check if the simulation is paused.

        Returns:
            bool: True if simulation is paused, False otherwise.
        """
        if not self._check_simulation_status():
            return False
        return bool(self.redis_client.exists(f"simulation:{self.simulation_uuid}:paused"))

    def _handle_control_command(self, command, simulator):
        """Handle simulation control commands.
        
        Args:
            command (str): The control command to execute.
            simulator (Simulator): The simulator object.
        """
        if command == "pause":
            self.redis_client.set(f"simulation:{self.simulation_uuid}:paused", "1")
            self.logger.info("Simulation paused")
        elif command == "resume":
            self.redis_client.delete(f"simulation:{self.simulation_uuid}:paused")
            self.logger.info("Simulation resumed")
        elif command == "stop":
            self.logger.info("Stopping simulation")
            simulator.running = False
        # Add more control command handling logic as needed

    def _write_simulation_state(self, simulator):
        """Write the current simulation state to Redis.

        Args:
            simulator (Simulator): The simulator object.
        """
        if not self._check_simulation_status():
            return
        try:
            simulation_state = SimulationState()
            simulation_state.simulation_time = traci.simulation.getTime()

            # Get all agent IDs
            vehicle_ids = traci.vehicle.getIDList()
            vru_ids = traci.person.getIDList()
            simulation_state.agent_count = {
                "vehicle": len(vehicle_ids),
                "vru": len(vru_ids),
            }

            # Add vehicle states
            vehicles = {}
            for vid in vehicle_ids:
                vehicle_state = AgentStateSimplified()
                vehicle_state.x,vehicle_state.y,vehicle_state.z = traci.vehicle.getPosition3D(vid)
                vehicle_state.orientation = traci.vehicle.getAngle(vid)
                vehicle_state.speed = traci.vehicle.getSpeed(vid)
                vehicle_state.length = traci.vehicle.getLength(vid)
                vehicle_state.width = traci.vehicle.getWidth(vid)
                vehicle_state.height = traci.vehicle.getHeight(vid)
                vehicle_state.type = traci.vehicle.getTypeID(vid)
                vehicles[vid] = vehicle_state

            simulation_state.agent_details["vehicle"] = vehicles

            # Add VRU states
            vrus = {}
            for vru_id in vru_ids:
                vru_state = AgentStateSimplified()
                vru_state.x, vru_state.y, vru_state.z = traci.person.getPosition3D(vru_id)
                vru_state.orientation = traci.person.getAngle(vru_id)
                vru_state.speed = traci.person.getSpeed(vru_id)
                vru_state.length = traci.person.getLength(vru_id)
                vru_state.width = traci.person.getWidth(vru_id)
                vru_state.height = traci.person.getHeight(vru_id)
                vru_state.type = traci.person.getTypeID(vru_id)
                vrus[vru_id] = vru_state

            simulation_state.agent_details["vru"] = vrus

            # Add traffic light states
            traffic_lights = {}
            for tl_id in traci.trafficlight.getIDList():
                sumo_signal = SUMOSignal()
                sumo_signal.x, sumo_signal.y = 0,0
                sumo_signal.tls = traci.trafficlight.getRedYellowGreenState(tl_id)
                traffic_lights[tl_id] = sumo_signal

            simulation_state.traffic_light_details = traffic_lights
            
            # Write to Redis with expiration
            self.redis_client.set(
                f"simulation:{self.simulation_uuid}:state", simulation_state.model_dump_json()
            )
            self.redis_client.expire(
                f"simulation:{self.simulation_uuid}:state", self.key_expiry
            )

        except Exception as e:
            self.logger.error(f"Error writing simulation state: {e}")

    def _handle_agent_command(self, command_data):
        """Handle agent control commands.
        
        Args:
            command_data (str): The agent command data.
        """
        try:
            command = AgentCommand.model_validate_json(command_data.decode("utf-8"))
            if command.agent_id != '':
                if command.agent_type not in ["vehicle", "vru"]:
                    self.logger.error(f"Invalid agent type: {command.agent_type}")
                    return False
                if command.command_type == "set_state":
                    if "position" in command.data:
                        x, y = command.data["position"]
                        if command.agent_type == "vehicle":
                            traci.vehicle.moveToXY(
                                command.agent_id, "", 0, x, y, command.data.get("angle", 0), 2
                            )

                            if "speed" in command.data:
                                traci.vehicle.setSpeed(command.agent_id, command.data["speed"])
                        else:
                            traci.person.moveToXY(
                                command.agent_id, "", x, y, command.data.get("angle", 0), 2
                            )

                            if "speed" in command.data:
                                traci.person.setSpeed(command.agent_id, command.data["speed"])

                self.logger.info(f"Agent command executed: {command_data}")
                return True

        except Exception as e:
            self.logger.error(f"Error handling agent command: {e}")
            return False

    def _reconnect_redis(self):
        """Reconnect to Redis server.

        Returns:
            bool: True if reconnection was successful, False otherwise.
        """
        try:
            self.logger.info("Attempting to reconnect to Redis...")
            self.redis_client = redis.Redis(**self.redis_config)
            self.logger.info("Successfully reconnected to Redis")
            return True
        except RedisError as e:
            self.logger.error(f"Failed to reconnect to Redis: {e}")
            return False

    def _handle_pending_agent_commands(self):
        """Handle all pending agent commands in the queue."""
        if not self._check_simulation_status():
            return
        """Handle all pending agent commands in the queue"""
        try:
            # Process up to 100 commands per step to prevent infinite loops
            for _ in range(100):
                command_data = self.redis_client.lpop(
                    f"simulation:{self.simulation_uuid}:agent_commands"
                )
                if not command_data:
                    break

                self._handle_agent_command(command_data)
        except Exception as e:
            self.logger.error(f"Error handling pending agent commands: {e}")


class TeraSimCoSimPluginBefore(TeraSimCoSimPlugin):
    def __init__(self, simulation_uuid, plugin_config = COSIM_PLUGIN_BEFORE_CONFIG, redis_config = DEFAULT_REDIS_CONFIG, key_expiry=3600, auto_run=False):
        super().__init__(simulation_uuid, plugin_config, redis_config, key_expiry, auto_run)

    def on_start(self, simulator, ctx):
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(**self.redis_config)

            # Clear old data and set initial state with expiration
            self.redis_client.delete(f"simulation:{self.simulation_uuid}:*")
            self.redis_client.set(
                f"simulation:{self.simulation_uuid}:status", "started", ex=self.key_expiry
            )

            self.logger.info(
                f"Redis connection established. Simulation UUID: {self.simulation_uuid}, start initialization!"
            )

            # Add this line to write initial simulation state
            # self._write_simulation_state(simulator)

            return True
        except RedisError as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error during initialization: {e}")
            return False
    
    def on_step(self, simulator, ctx):
        """Handle simulation step logic, including handling simulation level commands, handling agent-level command, and retrieving simulation states.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        
        Returns:
            bool: True if the simulation step was successful, False otherwise.
        """
        while True:
            # Handle simulation control commands
            command = self._get_and_handle_command(simulator)
            if command == "stop":
                return False

            # Handle all pending vehicle commands
            self._handle_pending_agent_commands()

            # Write current simulation state
            self._write_simulation_state(simulator)

            if self._is_simulation_paused():
                time.sleep(0.1)  # Wait while paused
                continue

            if not self.auto_run:
                if command == "tick":
                    break  # Proceed with the simulation step
                else:
                    time.sleep(0.005)  # Short sleep to prevent busy waiting
                    continue

            break  # Proceed with the simulation step in auto_run mode
        self.redis_client.set(
            f"simulation:{self.simulation_uuid}:status", "running", ex=self.key_expiry
        )
        print("Simulation step started")
        return True
    
    def on_stop(self, simulator, ctx):
        """Stop the simulation and clean up all simulation related data.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        try:
            if self.redis_client:
                # Set simulation end status briefly before cleanup
                self.redis_client.set(
                    f"simulation:{self.simulation_uuid}:status",
                    "finished",
                    ex=10,  # Keep status for 10 seconds only
                )

                # Clean up all simulation related data
                keys_pattern = f"simulation:{self.simulation_uuid}:*"
                for key in self.redis_client.scan_iter(match=keys_pattern):
                    self.redis_client.delete(key)

                # Close Redis connection
                self.redis_client.close()
                self.logger.info(
                    f"Simulation {self.simulation_uuid} stopped and data cleaned up"
                )
        except RedisError as e:
            self.logger.error(f"Error during Redis cleanup: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error during cleanup: {e}")
    

class TeraSimCoSimPluginAfter(TeraSimCoSimPlugin):
    def __init__(self, simulation_uuid, plugin_config = COSIM_PLUGIN_AFTER_CONFIG, redis_config = DEFAULT_REDIS_CONFIG, key_expiry=3600, auto_run=False):
        super().__init__(simulation_uuid, plugin_config, redis_config, key_expiry, auto_run)
    
    def on_start(self, simulator, ctx):
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(**self.redis_config)

            # Set initial state with expiration
            self.redis_client.set(
                f"simulation:{self.simulation_uuid}:status", "initialized", ex=self.key_expiry
            )

            self.logger.info(
                f"Redis connection established. Simulation UUID: {self.simulation_uuid}, finish initialization!"
            )

            # Add this line to write initial simulation state
            # self._write_simulation_state(simulator)

            return True
        except RedisError as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error during initialization: {e}")
            return False
    
    def on_step(self, simulator, ctx):
        self.redis_client.set(
            f"simulation:{self.simulation_uuid}:status", "stepped", ex=self.key_expiry
        )
        print("Simulation step finished!")
        return True
