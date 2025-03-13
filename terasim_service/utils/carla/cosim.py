import time
import redis
import math
import carla
import random

from .tools import (
    create_bike_blueprint,
    create_motor_blueprint,
    create_pedestrian_blueprint,
    create_vehicle_blueprint,
    destroy_all_actors,
    get_actor_id_from_attribute,
    sumo_to_carla,
    spawn_actor,
)
from ..service import (
    start_terasim,
    stop_terasim,
    tick_terasim,
    get_terasim_status,
    get_terasim_states,
)


class CarlaCosim(object):
    def __init__(self, args):
        self.args = args

        self.client = carla.Client(args.carla_host, args.carla_port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()

        self.traffic_lights = self.world.get_actors().filter("traffic.traffic_light")
        for traffic_light in self.traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Off)
            traffic_light.freeze(True)

        self.control_cav = args.control_cav
        self.async_mode = args.async_mode
        self.step_length = args.step_length

        self.vehicle_blueprints = create_vehicle_blueprint(self.world)
        self.motor_blueprints = create_motor_blueprint(self.world)
        self.pedestrian_blueprints = create_pedestrian_blueprint(self.world)
        self.bike_blueprints = create_bike_blueprint(self.world)

        # self.sync_cosim_construction_zone_to_carla()

        # start TeraSim
        terasim_init_command = {
            "config_file": args.terasim_config,
            "auto_run": False,
        }
        self.terasim = start_terasim(args.terasim_host, args.terasim_port, terasim_init_command)
        while True:
            terasim_status = get_terasim_status(args.terasim_host, args.terasim_port, self.terasim["simulation_id"])
            if terasim_status.get("status", None) == "initialized":
                break
            time.sleep(0.1)

    def tick(self):
        if self.async_mode:
            time_start = time.time()
            if self.control_cav:
                self.sync_carla_cav_to_cosim()

            self.sync_cosim_actor_to_carla()
            self.sync_cosim_tls_to_carla()

            self.world.tick()
            time_end = time.time()
            elapsed = time_end - time_start
            if elapsed < self.step_length:
                time.sleep(self.step_length - elapsed)
        else:
            tick_terasim(self.args.terasim_host, self.args.terasim_port, self.terasim["simulation_id"])
            while True:
                terasim_status = get_terasim_status(self.args.terasim_host, self.args.terasim_port, self.terasim["simulation_id"])
                if terasim_status.get("status", None) == "stepped":
                    break
                time.sleep(0.1)
            
            if self.control_cav:
                self.sync_carla_cav_to_cosim()

            self.sync_cosim_actor_to_carla()

            self.world.tick()

    def sync_carla_cav_to_cosim(self):
        vehicle_status, carla_id = get_actor_id_from_attribute(self.world, "CAV")

        if not vehicle_status:
            print("CAV not found in Carla simulation. Exiting...")
            return

        CAV = self.world.get_actor(carla_id)
        transform = CAV.get_transform()
        draw_text(self.world, transform.location + carla.Location(z=2.5), "CAV")
        # draw_point(
        #     self.world,
        #     size=0.05,
        #     color=(255, 0, 0),
        #     location=transform.location + carla.Location(z=2.5),
        #     life_time=0,
        # )

        velocity = CAV.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5

        cav_info = ActorDict()
        cav_info.header.timestamp = time.time()

        x = transform.location.x
        y = transform.location.y

        utm_x, utm_y = carla_to_utm(x, y)
        height = get_z_offset(
            world=self.world,
            start_location=carla.Location(
                transform.location.x, transform.location.y, 300
            ),
            end_location=carla.Location(
                transform.location.x, transform.location.y, 200
            ),
        )

        cav_info.data["CAV"] = Actor(
            x=utm_x,
            y=utm_y,
            z=height,
            length=5.0,
            width=1.8,
            height=1.8,
            orientation=math.radians(-transform.rotation.yaw),
            speed_long=speed,
        )

        self.redis_client.set(CAV_INFO, cav_info)

    def sync_cosim_tls_to_carla(self):
        try:
            cosim_tls_info = self.redis_client.get(TLS_INFO)
        except:
            print("cosim_tls_info not available. Exiting...")
            return

        if cosim_tls_info:
            cosim_tls_data = cosim_tls_info.data

            for node in cosim_tls_data:
                node_tls = cosim_tls_data[node]
                carla_tls_node = TLS_NODES[node]

                for i in range(len(carla_tls_node)):
                    light_ids = carla_tls_node[i]

                    if light_ids:
                        for light_id in light_ids:
                            light_actor = self.world.get_actor(light_id)
                            light_state = node_tls.tls[i]

                            if light_state == "G" or light_state == "g":
                                light_actor.set_state(carla.TrafficLightState.Green)
                            elif light_state == "Y" or light_state == "y":
                                light_actor.set_state(carla.TrafficLightState.Yellow)
                            elif light_state == "R" or light_state == "r":
                                light_actor.set_state(carla.TrafficLightState.Red)

    def sync_cosim_actor_to_carla(self):
        """Update all actors in cosim to CARLA.
        """
        terasim_states = get_terasim_states(self.args.terasim_host, self.args.terasim_port, self.terasim["simulation_id"])

        if not terasim_states:
            print("terasim_states not available. Exiting...")
            return

        cosim_id_record = set()

        for veh_id in terasim_states["agent_details"]["vehicle"]:
            self._process_vehicle(veh_id, terasim_states["agent_details"]["vehicle"][veh_id], cosim_id_record)
        
        for vru_id in terasim_states["agent_details"]["vru"]:
            self._process_vru(vru_id, terasim_states["agent_details"]["vru"][vru_id], cosim_id_record)

        self._cleanup_actors("vehicle", "vehicle.*", cosim_id_record)
        self._cleanup_actors("pedestrian", "walker.pedestrian.*", cosim_id_record)

        # self.sync_cosim_tls_to_carla()

    def sync_cosim_construction_zone_to_carla(self):
        def add_interpolated_points(points, offset):
            """
            Interpolates additional points to ensure no two consecutive points
            after UTM transformation have a distance greater than the specified offset.
            """
            refined_points = []
            print("enter add_interpolated_points")
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                # p1 = utm_to_carla(points[i][0], points[i][1])
                # p2 = utm_to_carla(points[i + 1][0], points[i + 1][1])
                refined_points.append(p1)  # Add the current transformed point

                # Calculate the 2D distance between transformed points (x, y only)
                distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                if distance > offset:
                    # Add intermediate points
                    num_new_points = int(distance // offset)
                    for j in range(1, num_new_points + 1):
                        # Linear interpolation to find new points
                        new_x = p1[0] + j * (p2[0] - p1[0]) / (num_new_points + 1)
                        new_y = p1[1] + j * (p2[1] - p1[1]) / (num_new_points + 1)
                        refined_points.append((new_x, new_y))

            refined_points.append(points[-1])  # Add the last transformed point
            return refined_points

        try:
            construction_zone_info = self.redis_client.get(CONSTRUCTION_ZONE_INFO)
            if not construction_zone_info:
                print("construction_zone_info is None or empty")
                return
        except Exception as e:
            print(f"Error fetching construction zone info: {e}")
            return

        print("entering construction zone")
        if construction_zone_info:
            closed_lane_shapes = construction_zone_info.closed_lane_shapes

            for closed_lane_shape in closed_lane_shapes:
                closed_lane_shape = add_interpolated_points(closed_lane_shape, 10)
                for cone_point in closed_lane_shape:
                    construction_cone = create_construction_zone_blueprint(self.world)
                    spawn_point = carla.Transform()
                    spawn_point.location.x, spawn_point.location.y = utm_to_carla(
                        cone_point[0], cone_point[1]
                    )
                    spawn_point.location.z = get_z_offset(
                        self.world,
                        start_location=carla.Location(
                            spawn_point.location.x, spawn_point.location.y, 300
                        ),
                        end_location=carla.Location(
                            spawn_point.location.x, spawn_point.location.y, 200
                        ),
                    )
                    id = spawn_actor(
                        client=self.client,
                        blueprint=construction_cone,
                        transform=spawn_point,
                    )
                    print(f"created construction cone: {id}")

    def _process_vehicle(self, veh_id, veh_info, cosim_id_record):
        """Process a vehicle actor."""
        cosim_id_record.add(veh_id)

        sumo_location = [veh_info["x"], veh_info["y"], veh_info["z"]]
        sumo_rotation = [0.0, veh_info["orientation"], 0.0]
        shape = [veh_info["length"], veh_info["width"], veh_info["height"]]

        vehicle_status, carla_id = get_actor_id_from_attribute(self.world, veh_id)

        if not vehicle_status:
            blueprint = random.choice(
                self.motor_blueprints
                if "BIKE" in veh_info["type"]
                else self.vehicle_blueprints
            )
            blueprint.set_attribute("role_name", veh_id)
            blueprint.set_attribute("color", "0, 102, 204")
            sumo_offset = [0.0, 0.0, shape[2]] # spawn the vehicle higher than the ground to make sure it is available
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            carla_id = spawn_actor(self.client, blueprint, carla_trasform)
            print(f"Spawned vehicle in CARLA: SUMO ID {veh_id}, CARLA ID {carla_id}")
        else:
            sumo_offset = [0.0, 0.0, 0.0] # move the vehicle back to the ground
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            vehicle = self.world.get_actor(carla_id)
            vehicle.set_transform(carla_trasform)

    def _process_vru(self, vru_id, vru_info, cosim_id_record):
        """Process a pedestrian actor."""
        cosim_id_record.add(vru_id)

        sumo_location = [vru_info["x"], vru_info["y"], vru_info["z"]]
        sumo_rotation = [0.0, vru_info["orientation"], 0.0]
        shape = [vru_info["length"], vru_info["width"], vru_info["height"]]

        vru_status, carla_id = get_actor_id_from_attribute(self.world, vru_id)

        if not vru_status:
            blueprint = random.choice(
                self.bike_blueprints
                if "BIKE" in vru_info["type"]
                else self.pedestrian_blueprints
            )
            blueprint.set_attribute("role_name", vru_id)
            sumo_offset = [0.0, 0.0, shape[2]] # spawn the VRU higher than the ground to make sure it is available
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            carla_id = spawn_actor(self.client, blueprint, carla_trasform)
            print(f"Spawned pedestrian in CARLA: SUMO ID {vru_id}, CARLA ID {carla_id}")
        else:
            # move the VRU back to the ground
            sumo_offset = [0.0, 0.0, shape[2]/2.0]
            if "BIKE" in vru_info["type"]:
                sumo_offset = [0.0, 0.0, 0.0]
            carla_trasform = sumo_to_carla(sumo_location, sumo_rotation, shape, sumo_offset)
            pedestrian = self.world.get_actor(carla_id)
            pedestrian.set_transform(carla_trasform)

        if carla_id > 0:
            if "BIKE" not in vru_info["type"]:
                radians = math.radians(90 - vru_info["orientation"])
                orientation = math.atan2(math.sin(radians), math.cos(radians))
                direction_x, direction_y = math.cos(orientation), math.sin(orientation)
                walker_control = carla.WalkerControl(
                    direction=carla.Vector3D(
                        direction_x, direction_y, 0
                    ),
                    speed=vru_info["speed"],
                )
                self.world.get_actor(carla_id).apply_control(walker_control)
            else:
                control = carla.VehicleControl()
                self.world.get_actor(carla_id).apply_control(control)

    def _cleanup_actors(self, actor_type, pattern, cosim_id_record):
        """Clean up CARLA actors not in the cosim actor record."""
        actors_to_destroy = [
            actor
            for actor in self.world.get_actors().filter(pattern)
            if actor.attributes.get("role_name") not in cosim_id_record
            and actor.attributes.get("role_name") != "CAV"
        ]

        for actor in actors_to_destroy:
            actor.destroy()
            print(f"Destroyed {actor_type}: {actor.attributes.get('role_name')}")

    def close(self):
        """
        Cleans synchronization and resets the simulation settings.
        """
        # Configuring carla simulation in async mode.
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        destroy_all_actors(self.world)
