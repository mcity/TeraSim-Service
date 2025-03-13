import argparse
import carla

from terasim_service.utils.carla import CarlaCosim


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--carla_host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server for Carla (default: 127.0.0.1)')
    argparser.add_argument(
        '--carla_port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to for Carla (default: 2000)')
    argparser.add_argument(
        '-s', '--step_length',
        metavar='P',
        default=0.04,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--control_cav',
        action='store_true',
        help='Activate CAV manual control mode execution')
    argparser.add_argument(
        '--async_mode',
        action='store_true',
        help='Activate async mode execution')
    argparser.add_argument(
        '--terasim_host',
        default='localhost',
        help='IP of the host server for TeraSim (default: localhost)')
    argparser.add_argument(
        '--terasim_port',
        default=8000,
        type=int,
        help='TCP port to listen to for TeraSim (default: 8000)')
    argparser.add_argument(
        '--terasim_config',
        default='examples/simulation_config.yaml',
        help='Configuation file path for TeraSim (default: examples/simulation_config.yaml)')
    args = argparser.parse_args()
    carla_cosim = CarlaCosim(args)

    settings = carla_cosim.world.get_settings()
    settings.fixed_delta_seconds = args.step_length
    settings.synchronous_mode = True
    carla_cosim.world.apply_settings(settings)

    carla_cosim.world.set_weather(carla.WeatherParameters.WetSunset)

    try:
        while True:
            carla_cosim.tick()

    except KeyboardInterrupt:
        print("Cancelled by user.")

    finally:
        print("Cleaning synchronization")
        carla_cosim.close()


if __name__ == "__main__":
    main()