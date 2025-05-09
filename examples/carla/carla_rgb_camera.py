import cv2
import time
import carla
import numpy as np
import threading


# CARLA sensor manager class
class SensorManager:
    """Manages CARLA RGB camera sensor and processes data."""

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
            int(sensor_options.get("image_size_x", 1920)),
            int(sensor_options.get("image_size_y", 1080)),
        )
        if self.video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_path, fourcc, self.fps, self.frame_size
            )

    def init_sensor(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        for key, value in sensor_options.items():
            camera_bp.set_attribute(key, value)

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self.camera_callback)

        return camera

    def camera_callback(self, image):
        """Callback function for the RGB camera."""
        # Convert the raw image data to a numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(
            array, (image.height, image.width, 4)
        )  # BGRA format from CARLA
        img_bgr = array[:, :, :3]  # Remove alpha channel (BGR format)

        # Store the latest image in a thread-safe manner
        with self.lock:
            self.latest_image = img_bgr
            if self.video_writer is not None:
                self.video_writer.write(img_bgr)

    def get_latest_image(self):
        """Get the latest image in a thread-safe manner."""
        with self.lock:
            return self.latest_image

    def export_video(self):
        """Release the video writer if used."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def destroy(self):
        """Clean up the sensor."""
        if self.sensor is not None:
            self.sensor.destroy()
        self.export_video()
        cv2.destroyAllWindows()


def main():
    # Connect to the CARLA server
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)
    world = client.get_world()

    # Find the vehicle with the role_name "AV"
    AV = None
    while AV is None:
        actor_list = world.get_actors().filter("vehicle.*")
        for x in actor_list:
            if x.attributes.get("role_name") == "AV":
                AV = x
        print("AV not found. Waiting for AV to spawn...")
        time.sleep(0.1)

    print("Found AV, attaching RGB camera...")

    # Create and attach the RGB camera
    sensor_manager = SensorManager(
        world,
        carla.Transform(
            carla.Location(x=-11.0, y=0.0, z=7.0),  # Adjust camera position
            carla.Rotation(roll=0.0, pitch=-25.0, yaw=0.0),  # Adjust camera rotation
        ),
        AV,
        {
            "image_size_x": "1920",
            "image_size_y": "1080",
        },
        video_path="output.mp4",  # Path to save the video
        fps=10,  # Frames per second for the video
    )

    try:
        while True:
            time.sleep(0.01)
            print("Running...")

            # Get the latest image from the sensor manager
            img_bgr = sensor_manager.get_latest_image()
            if img_bgr is not None:
                # Display the image using OpenCV
                cv2.imshow("RGB Camera", img_bgr)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        sensor_manager.destroy()


if __name__ == "__main__":
    main()
