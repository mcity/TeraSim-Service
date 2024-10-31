import os
import math
import time
import sumolib
import lxml.etree as ET
import redis
from redis.exceptions import RedisError
import uuid
import logging
from logging.handlers import RotatingFileHandler
import json
from terasim.overlay import traci
from terasim.simulator import Simulator


class TeraSimControlPlugin:

    def __init__(
        self,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        simulation_uuid=None,
        key_expiry=3600,
        auto_run=False,
    ):
        # Redis connection configuration
        self.redis_config = {"host": redis_host, "port": redis_port, "db": redis_db}
        self.redis_client = None
        # Unique identifier for this simulation instance
        self.simulation_uuid = simulation_uuid or str(uuid.uuid4())
        # Key expiration time in seconds (default: 1 hour)
        self.key_expiry = key_expiry
        self.auto_run = auto_run

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"TeraSimControlPlugin-{self.simulation_uuid}")
        logger.setLevel(logging.DEBUG)

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            f"terasim_control_plugin_{self.simulation_uuid}.log",
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
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(**self.redis_config)

            # Clear old data and set initial state with expiration
            self.redis_client.delete(f"sim:{self.simulation_uuid}:*")
            self.redis_client.set(
                f"sim:{self.simulation_uuid}:status", "started", ex=self.key_expiry
            )

            self.logger.info(
                f"Redis connection established. Simulation UUID: {self.simulation_uuid}"
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

    def on_step(self, simulator: Simulator, ctx):
        while True:
            # Handle simulation control commands
            command = self._get_and_handle_command(simulator)
            if command == "stop":
                return True

            # Handle all pending vehicle commands
            self._handle_pending_vehicle_commands()

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

        return True

    def on_stop(self, simulator: Simulator, ctx):
        try:
            if self.redis_client:
                # Set simulation end status briefly before cleanup
                self.redis_client.set(
                    f"sim:{self.simulation_uuid}:status",
                    "finished",
                    ex=10,  # Keep status for 10 seconds only
                )

                # Clean up all simulation related data
                keys_pattern = f"sim:{self.simulation_uuid}:*"
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

    def inject(self, simulator: Simulator, ctx):
        self.ctx = ctx
        self.simulator = simulator

        # Inject hooks into the simulator pipeline
        simulator.start_pipeline.hook("control_start", self.on_start, priority=-90)
        simulator.step_pipeline.hook("control_step", self.on_step, priority=-90)
        simulator.stop_pipeline.hook("control_stop", self.on_stop, priority=-90)

    def _check_simulation_status(self) -> bool:
        """Check if simulation is still running
        Returns:
            bool: True if simulation is running, False if stopped or doesn't exist
        """
        status = self.redis_client.get(f"sim:{self.simulation_uuid}:status")
        if not status or status.decode("utf-8") == "finished":
            self.logger.warning(
                f"Simulation {self.simulation_uuid} is stopped or doesn't exist"
            )
            return False
        return True

    def _get_and_handle_command(self, simulator: Simulator) -> str | None:
        if not self._check_simulation_status():
            return "stop"
        command = self.redis_client.get(f"sim:{self.simulation_uuid}:control")
        if command:
            command = command.decode("utf-8")
            self._handle_control_command(command, simulator)
            if command != "stop":
                self.redis_client.delete(f"sim:{self.simulation_uuid}:control")
        return command

    def _is_simulation_paused(self) -> bool:
        if not self._check_simulation_status():
            return False
        return bool(self.redis_client.exists(f"sim:{self.simulation_uuid}:paused"))

    def _handle_control_command(self, command, simulator):
        if command == "pause":
            self.redis_client.set(f"sim:{self.simulation_uuid}:paused", "1")
            self.logger.info("Simulation paused")
        elif command == "resume":
            self.redis_client.delete(f"sim:{self.simulation_uuid}:paused")
            self.logger.info("Simulation resumed")
        elif command == "stop":
            self.logger.info("Stopping simulation")
            simulator.soft_request_stop()
        # Add more control command handling logic as needed

    def _write_simulation_state(self, simulator):
        if not self._check_simulation_status():
            return
        try:
            # Get all vehicle IDs
            vehicle_ids = traci.vehicle.getIDList()

            # Basic simulation state
            state = {
                "vehicle_count": len(vehicle_ids),
                "simulation_time": traci.simulation.getTime(),
            }

            # Add vehicle states
            vehicles = {}
            for vid in vehicle_ids:
                vehicles[vid] = {
                    "position": traci.vehicle.getPosition(vid),  # (x,y)
                    "speed": traci.vehicle.getSpeed(vid),
                    "angle": traci.vehicle.getAngle(vid),
                }

            state["vehicles"] = vehicles

            # Write to Redis with expiration
            self.redis_client.hset(
                f"sim:{self.simulation_uuid}:state", mapping={"data": json.dumps(state)}
            )
            self.redis_client.expire(
                f"sim:{self.simulation_uuid}:state", self.key_expiry
            )

        except Exception as e:
            self.logger.error(f"Error writing simulation state: {e}")

    def _handle_vehicle_command(self, command_data):
        """Handle vehicle control commands"""
        try:
            vehicle_id = command_data["vehicle_id"]
            command_type = command_data["type"]

            if command_type == "set_state":
                if "position" in command_data:
                    x, y = command_data["position"]
                    traci.vehicle.moveToXY(
                        vehicle_id, "", 0, x, y, command_data.get("angle", 0), 2
                    )

                if "speed" in command_data:
                    traci.vehicle.setSpeed(vehicle_id, command_data["speed"])

                if "angle" in command_data:
                    traci.vehicle.setAngle(vehicle_id, command_data["angle"])

            self.logger.info(f"Vehicle command executed: {command_data}")
            return True

        except Exception as e:
            self.logger.error(f"Error handling vehicle command: {e}")
            return False

    def _reconnect_redis(self):
        try:
            self.logger.info("Attempting to reconnect to Redis...")
            self.redis_client = redis.Redis(**self.redis_config)
            self.logger.info("Successfully reconnected to Redis")
            return True
        except RedisError as e:
            self.logger.error(f"Failed to reconnect to Redis: {e}")
            return False

    def _handle_pending_vehicle_commands(self):
        if not self._check_simulation_status():
            return
        """Handle all pending vehicle commands in the queue"""
        try:
            # Process up to 100 commands per step to prevent infinite loops
            for _ in range(100):
                command_data = self.redis_client.lpop(
                    f"sim:{self.simulation_uuid}:vehicle_commands"
                )
                if not command_data:
                    break

                command = json.loads(command_data.decode("utf-8"))
                self._handle_vehicle_command(command)
        except Exception as e:
            self.logger.error(f"Error handling pending vehicle commands: {e}")
