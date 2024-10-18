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
            return True
        except RedisError as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error during initialization: {e}")
            return False

    def on_step(self, simulator: Simulator, ctx):
        try:
            while True:
                # Check if simulation is paused
                if self.redis_client.get(f"sim:{self.simulation_uuid}:paused"):
                    time.sleep(0.1)  # Wait while paused
                    continue

                if not self.auto_run:
                    command = self.redis_client.get(
                        f"sim:{self.simulation_uuid}:control"
                    )
                    if command and command.decode("utf-8") == "tick":
                        # clear tick command
                        self.redis_client.delete(f"sim:{self.simulation_uuid}:control")
                    elif command:
                        # handle other commands (currently only "stop")
                        self._handle_control_command(command, simulator)
                        continue
                    elif not command:
                        time.sleep(0.005)  # short sleep to prevent busy waiting
                        continue

                # continue simulation step
                break

            return True
        except RedisError as e:
            self.logger.error(f"Redis error during step: {e}")
            if self._reconnect_redis():
                return True
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error during simulation step: {e}")
            return False

    def on_stop(self, simulator: Simulator, ctx):
        try:
            if self.redis_client:
                # Set simulation end status with expiration
                self.redis_client.set(
                    f"sim:{self.simulation_uuid}:status", "finished", ex=self.key_expiry
                )
                # Close Redis connection
                self.redis_client.close()
                self.logger.info("Redis connection closed.")
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

    def _handle_control_command(self, command, simulator):
        command = command.decode("utf-8")
        if command == "pause":
            self.redis_client.set(f"sim:{self.simulation_uuid}:paused", "1")
            self.logger.info("Simulation paused")
        elif command == "resume":
            self.redis_client.delete(f"sim:{self.simulation_uuid}:paused")
            self.logger.info("Simulation resumed")
        elif command == "stop":
            simulator.stop()
            self.logger.info("Simulation stopped")
        # Add more control command handling logic as needed
        self.redis_client.delete(f"sim:{self.simulation_uuid}:control")

    def _write_simulation_state(self, simulator):
        # Write simulation state to Redis with expiration
        vehicle_count = len(traci.vehicle.getIDList())
        self.redis_client.hset(
            f"sim:{self.simulation_uuid}:state",
            mapping={
                "step": simulator.step,
                "vehicle_count": vehicle_count,
                "simulation_time": simulator.simulation_time,
            },
        )
        self.redis_client.expire(f"sim:{self.simulation_uuid}:state", self.key_expiry)

    def _reconnect_redis(self):
        try:
            self.logger.info("Attempting to reconnect to Redis...")
            self.redis_client = redis.Redis(**self.redis_config)
            self.logger.info("Successfully reconnected to Redis")
            return True
        except RedisError as e:
            self.logger.error(f"Failed to reconnect to Redis: {e}")
            return False
