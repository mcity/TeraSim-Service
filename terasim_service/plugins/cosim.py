import json
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
import redis
from redis.exceptions import RedisError
import time

from terasim.overlay import traci
from terasim.simulator import Simulator

from terasim_nde_nade.adversity import ConstructionAdversity

from .base import BasePlugin, DEFAULT_REDIS_CONFIG

from ..utils import SimulationState, AgentStateSimplified, SUMOSignal, AgentCommand


def interpolate_by_distance(points, step):
    """
    Interpolate a tuple of tuples so that the distance between each point is equal to 'step'.

    Args:
        points (tuple of tuple): Original shape, e.g., ((x1, y1), (x2, y2), ...)
        step (float): Desired distance between points.

    Returns:
        list of list: Interpolated points as [[x, y], ...] with equal spacing.
    """
    points = np.array(points, dtype=np.float32)
    # Compute distances between consecutive points
    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.insert(np.cumsum(seg_lengths), 0, 0)
    total_length = cumulative[-1]
    if total_length == 0:
        return [points[0].tolist()]
    # Generate equally spaced distances
    num_points = int(np.floor(total_length / step)) + 1
    distances = np.linspace(0, total_length, num_points)
    # Interpolate x and y separately
    x_interp = np.interp(distances, cumulative, points[:, 0])
    y_interp = np.interp(distances, cumulative, points[:, 1])
    return [[float(x), float(y)] for x, y in zip(x_interp, y_interp)]


def generate_construction_zone_shape(lane_shape, lane_width, direction):
    """
    Generate a construction zone shape based on the lane shape and lane width.
    The first ten points of the lane_shape are offset laterally, with the offset
    gradually changing from direction * lane_width/2 to -direction * lane_width/2.
    The remaining points are offset by a constant -direction * lane_width/2.

    Args:
        lane_shape (list of list): The lane shape as a list of [x, y] points.
        lane_width (float): The width of the lane.
        direction (int): -1 for from left to right, 1 for from right to left.

    Returns:
        list of list: The offset lane shape.
    """
    n = min(10, len(lane_shape))
    construction_zone_shape = []
    for i, pt in enumerate(lane_shape):
        pt = np.array(pt)
        # Compute tangent direction
        if i < len(lane_shape) - 1:
            next_pt = np.array(lane_shape[i + 1])
            dir_vec = next_pt - pt
        else:
            prev_pt = np.array(lane_shape[i - 1])
            dir_vec = pt - prev_pt
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            dir_vec = np.array([1.0, 0.0])
        else:
            dir_vec = dir_vec / norm
        # Normal vector (perpendicular)
        normal = np.array([-dir_vec[1], dir_vec[0]]) * direction * -1

        # Compute offset
        if i < n:
            # Linear interpolation from +lane_width/2 to -lane_width/2
            alpha = i / (n - 1) if n > 1 else 0
            offset_val = (1 - alpha) * (lane_width / 2) + alpha * (-lane_width / 2)
        else:
            offset_val = - lane_width / 2

        offset_pt = pt + normal * offset_val
        construction_zone_shape.append(offset_pt.tolist())
    return construction_zone_shape


DEFAULT_COSIM_PLUGIN_CONFIG = {
    "name": "terasim_cosim_plugin",
    "priority": {
        "before_env": {
            "start": -90,
            "step": -90,
            "stop": -90,
        },
        "after_env": {
            "start": 90,
            "step": 90,
            "stop": 90,
        },
    },
}


class TeraSimCoSimPlugin(BasePlugin):
    def __init__(
        self,
        simulation_uuid: str,
        plugin_config: dict = DEFAULT_COSIM_PLUGIN_CONFIG,
        redis_config: dict = DEFAULT_REDIS_CONFIG,
        base_dir: str = "output",
        key_expiry=3600,
        auto_run=False,
    ):
        """Initialize the Co-Simulation plugin.

        Args:
            simulation_uuid (str): Unique identifier for the simulation instance.
            plugin_config (dict, optional): Configuration for the plugin. Defaults to DEFAULT_COSIM_PLUGIN_CONFIG.
            redis_config (dict, optional): Configuration for the Redis connection. Defaults to DEFAULT_REDIS_CONFIG.
            base_dir (str, optional): Base directory for the log file. Defaults to "output".
            key_expiry (int, optional): Key expiration time in seconds. Defaults to 3600.
            auto_run (bool, optional): Flag to enable auto-run mode. Defaults to False.
        """
        super().__init__(simulation_uuid, plugin_config, redis_config)
        # Key expiration time in seconds (default: 1 hour)
        self.key_expiry = key_expiry
        self.auto_run = auto_run

        # Setup logging
        self.logger = self._setup_logger(base_dir)

        # Maintain controlled agents in each step, assuming each agent can be controlled by only one command
        self.controlled_agents_each_step = set()

        # Cache construction zone shapes
        self.construction_zone_shapes = None

    def _setup_logger(self, base_dir: str) -> logging.Logger:
        """Setup logger for the plugin.

        Args:
            base_dir (str): Base directory for the log file.

        Returns:
            logging.Logger: Logger instance for the plugin.
        """
        logger = logging.getLogger(f"{self.plugin_name}-{self.simulation_uuid}")
        logger.setLevel(logging.DEBUG)

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            f"{base_dir}/{self.plugin_name}.log",
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

    def function_before_env_start(self, simulator: Simulator, ctx):
        """Connect to the Redis server and set the simulation status to be 'initializing'.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(**self.redis_config)

            # Clear old data and set initial state with expiration
            self.redis_client.delete(f"simulation:{self.simulation_uuid}:*")
            self.redis_client.set(
                f"simulation:{self.simulation_uuid}:status", "initializing", ex=self.key_expiry
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
        
    def function_after_env_start(self, simulator: Simulator, ctx):
        """Set the simulation status to 'wait_for_tick' after finishing the intialization.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        try:
            # Set initial state with expiration
            self.redis_client.set(
                f"simulation:{self.simulation_uuid}:status", "wait_for_tick", ex=self.key_expiry
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

    def function_before_env_step(self, simulator: Simulator, ctx):
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
            self.controlled_agents_each_step.clear()
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
        self.logger.info("Simulation step started")
        return True
    
    def function_after_env_step(self, simulator: Simulator, ctx):
        """Handle post-simulation step logic, including updating simulation status.
        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        Returns:
            bool: True if the simulation step was successful, False otherwise.
        """
        self.redis_client.set(
            f"simulation:{self.simulation_uuid}:status", "ticked", ex=self.key_expiry
        )
        self.logger.info("Simulation step finished!")
        return True

    def function_before_env_stop(self, simulator: Simulator, ctx):
        """Handle simulation stopping logic. Default implementation does nothing.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        pass

    def function_after_env_stop(self, simulator: Simulator, ctx):
        """Handle post-simulation stopping logic, including updating simulation status.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        try:
            if self.redis_client:
                finish_string = f"Simulation {self.simulation_uuid} finished!"
                # Set simulation end status briefly before cleanup
                results_dict = {
                    "finish_reason": simulator.env.record.get("finish_reason",""),
                    "collider": simulator.env.record.get("collider",""),
                    "victim": simulator.env.record.get("victim",""),
                }
                results_str = json.dumps(results_dict)
                self.redis_client.set(
                    f"simulation:{self.simulation_uuid}:status",
                    "finished",
                    ex=10,  # Keep status for 10 seconds only
                )
                self.redis_client.set(
                    f"simulation:{self.simulation_uuid}:result",
                    results_str,
                    ex=1800,  # Keep status for 30 minutes
                )

                # Clean up all simulation related data, except status
                keys_pattern = f"simulation:{self.simulation_uuid}:*"
                status_key = f"simulation:{self.simulation_uuid}:status"
                result_key = f"simulation:{self.simulation_uuid}:result"
                for key in self.redis_client.scan_iter(match=keys_pattern):
                    if key.decode() != status_key and key.decode() != result_key:
                        # Delete all keys except status and result
                        self.redis_client.delete(key)

                # Close Redis connection
                self.redis_client.close()
                self.logger.info(finish_string)
        except RedisError as e:
            self.logger.error(f"Error during Redis cleanup: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error during cleanup: {e}")
    
    def inject(self, simulator: Simulator, ctx):
        """Inject the plugin into the simulation.

        Args:
            simulator (Simulator): The simulator object.
            ctx (dict): The context information.
        """
        self.ctx = ctx
        self.simulator = simulator

        simulator.start_pipeline.hook(f"{self.plugin_name}_before_env_start", self.function_before_env_start, priority=self.plugin_priority["before_env"]["start"])
        simulator.start_pipeline.hook(f"{self.plugin_name}_after_env_start", self.function_after_env_start, priority=self.plugin_priority["after_env"]["start"])
        simulator.step_pipeline.hook(f"{self.plugin_name}_before_env_step", self.function_before_env_step, priority=self.plugin_priority["before_env"]["step"])
        simulator.step_pipeline.hook(f"{self.plugin_name}_after_env_step", self.function_after_env_step, priority=self.plugin_priority["after_env"]["step"])
        simulator.stop_pipeline.hook(f"{self.plugin_name}_before_env_stop", self.function_before_env_stop, priority=self.plugin_priority["before_env"]["stop"])
        simulator.stop_pipeline.hook(f"{self.plugin_name}_after_env_stop", self.function_after_env_stop, priority=self.plugin_priority["after_env"]["stop"])
    
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

            # Get all interested agent IDs
            if "centered_agent_ID" in self.plugin_config:
                centered_agent_ID = self.plugin_config["centered_agent_ID"]
                agent_ids = traci.vehicle.getContextSubscriptionResults(centered_agent_ID).keys()
                vehicle_ids = []
                vru_ids = []
                for _id in agent_ids:
                    if "BV" in _id or "AV" in _id:
                        vehicle_ids.append(_id)
                    elif "VRU" in _id:
                        vru_ids.append(_id)
            else:
                vehicle_ids = traci.vehicle.getIDList()
                vru_ids = traci.person.getIDList()
            # TODO: VRUs do not show up in the getContextSubscriptionResults, just use all VRUs at the current stage
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
                vehicle_state.lon,vehicle_state.lat = traci.simulation.convertGeo(vehicle_state.x, vehicle_state.y)
                vehicle_state.sumo_angle = traci.vehicle.getAngle(vid)
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
                vru_state.lon, vru_state.lat = traci.simulation.convertGeo(vru_state.x, vru_state.y)
                vru_state.sumo_angle = traci.person.getAngle(vru_id)
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
                tls_information = {
                    "programs": {}
                }
                tls = self.simulator.sumo_net.getTLS(tl_id)
                programs = tls.getPrograms()
                for program_id, program in programs.items():
                    # Get the program parameters
                    program_parameters = program.getParams()
                    tls_information["programs"][program_id] = {
                        "parameters": program_parameters
                    }
                sumo_signal.information = json.dumps(tls_information)
                traffic_lights[tl_id] = sumo_signal

            simulation_state.traffic_light_details = traffic_lights

            # Add construction zone shapes
            if self.construction_zone_shapes is None and simulator.env.static_adversity is not None and simulator.env.static_adversity.adversities is not None:
                self.construction_zone_shapes = {}
                for adversity in simulator.env.static_adversity.adversities:
                    if isinstance(adversity, ConstructionAdversity):
                        lane_shape = traci.lane.getShape(adversity._lane_id)
                        if lane_shape: # convert to list of lists
                            lane_shape = interpolate_by_distance(lane_shape, 2.0)
                            lane_index = int(adversity._lane_id.split("_")[-1])
                            edge_id = traci.lane.getEdgeID(adversity._lane_id)
                            if lane_index == 0:
                                # From right to left
                                direction = 1
                            elif lane_index == traci.edge.getLaneNumber(edge_id) - 1:
                                # From left to right
                                direction = -1
                            else:
                                # Middle lane, no construction zone
                                continue
                            construction_zone_shape = generate_construction_zone_shape(lane_shape, traci.lane.getWidth(adversity._lane_id), direction)
                            self.construction_zone_shapes[adversity._lane_id] = construction_zone_shape

            simulation_state.construction_zone_details = self.construction_zone_shapes
            
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
                if command.agent_id in self.controlled_agents_each_step:
                    self.logger.debug(f"Agent {command.agent_id} is already controlled")
                    return True
                self.controlled_agents_each_step.add(command.agent_id)
                if command.command_type == "set_state":
                    # Check that exactly one of position or lonlat is present
                    has_position = "position" in command.data
                    has_lonlat = "lonlat" in command.data
                    if not (has_position ^ has_lonlat):  # XOR operation ensures exactly one is True
                        self.logger.error("Must specify exactly one of position or lonlat")
                        return False
                    if "position" in command.data:
                        x, y = command.data["position"]
                    elif "lonlat" in command.data:
                        lon, lat = command.data["lonlat"]
                        x, y = traci.simulation.convertGeo(lon, lat, fromGeo=True)
                    if command.agent_type == "vehicle":
                        traci.vehicle.moveToXY(
                            command.agent_id, "", 0, x, y, command.data.get("sumo_angle", 0), 2
                        )

                        if "speed" in command.data:
                            traci.vehicle.setPreviousSpeed(command.agent_id, command.data["speed"])
                    else:
                        traci.person.moveToXY(
                            command.agent_id, "", x, y, command.data.get("sumo_angle", 0), 2
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


