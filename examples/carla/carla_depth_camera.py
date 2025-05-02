import cv2
import time
import carla
import numpy as np
import threading

class DepthSensorManager:
    """Manages CARLA depth camera sensor and processes data."""

    def __init__(self, world, transform, attached, sensor_options, video_path=None, fps=10):
        self.world = world
        self.sensor_options = sensor_options
        self.sensor = self.init_sensor(transform, attached, sensor_options)
        self.latest_image = None
        self.lock = threading.Lock()

        self.video_writer = None
        self.video_path = video_path
        self.fps = fps
        self.frame_size = (
            int(sensor_options.get("image_size_x", 800)),
            int(sensor_options.get("image_size_y", 600)),
        )
        if self.video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, self.frame_size, False
            )

    def init_sensor(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.depth")
        for key, value in sensor_options.items():
            camera_bp.set_attribute(key, value)
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.camera_callback)
        return camera

    def camera_callback(self, image):
        """Callback function for the depth camera."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # Extract B, G, R channels
        B = array[:, :, 0].astype(np.float32)
        G = array[:, :, 1].astype(np.float32)
        R = array[:, :, 2].astype(np.float32)
        # Apply depth conversion algorithm
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = 1000 * normalized

        # Set minimum and maximum depth values for clipping
        min_depth = 0.1
        max_depth = 255.0

        # Clip depth values to enhance contrast
        depth_clipped = np.clip(depth_in_meters, min_depth, max_depth)

        # Apply logarithmic scaling
        depth_log = np.log1p(depth_clipped)

        # Normalize to the 8-bit range
        depth_normalized = depth_log / np.max(depth_log)
        depth_image_8bit = np.uint8(depth_normalized * 255)
       

        with self.lock:
            self.latest_image = depth_image_8bit
            if self.video_writer is not None:
                self.video_writer.write(depth_image_8bit)

    def get_latest_image(self):
        with self.lock:
            return self.latest_image

    def export_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
        self.export_video()
        cv2.destroyAllWindows()

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Find the vehicle with the role_name "CAV"
    CAV = None
    while CAV is None:
        actor_list = world.get_actors().filter("vehicle.*")
        for x in actor_list:
            if x.attributes.get("role_name") == "CAV":
                CAV = x
        print("CAV not found. Waiting for CAV to spawn...")
        time.sleep(0.1)

    print("Found CAV, attaching Depth camera...")

    sensor_manager = DepthSensorManager(
        world,
        carla.Transform(
            carla.Location(x=0.5, y=0.0, z=1.5),
            carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0),
        ),
        CAV,
        {
            "image_size_x": "800",
            "image_size_y": "600",
        },
        video_path="depth_output.mp4",
        fps=10,
    )

    try:
        while True:
            time.sleep(0.01)
            print("Running...")

            img_depth = sensor_manager.get_latest_image()
            if img_depth is not None:
                cv2.imshow("Depth Camera", img_depth)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        sensor_manager.destroy()

if __name__ == "__main__":
    main()