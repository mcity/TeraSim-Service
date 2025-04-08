import carla
import pyproj
import utm


UTM_OFFSET = [-277600 + 102.89, -4686800 + 281.25, 0.0]
ZONE_NUMBER = 17
ZONE_LETTER = "T"
GNSS_ORIGIN = [42.3005934157, -83.699283188811]

CARLA_CLIENT = carla.Client("localhost", 2000)
CARLA_CLIENT.load_world("McityMap_Main")
CARLA_WORLD = CARLA_CLIENT.get_world()



def latlon_to_xy(lat_center, lon_center, lat_point, lon_point):
    # Define a Transverse Mercator projection with the center point as the origin
    projection = pyproj.Proj(
        proj="tmerc", lat_0=lat_center, lon_0=lon_center, ellps="WGS84"
    )

    # Convert the point lat/lon to x, y coordinates relative to the center point
    x, y = projection(lon_point, lat_point)

    return x, y


if __name__ == "__main__":
    # Define the path to the XML file
    # xml_file_path = "examples/maps/Mcity_safetest/mcity.net.xml"
    original_sumo_net_file_path = "TeraSim-Service/examples/maps/Mcity_safetest/mcity.net.xml"
    new_sumo_net_file_path = "TeraSim-Service/examples/maps/Mcity_safetest/mcity_new.net.xml"

    # Read the original XML file line by line
    with open(original_sumo_net_file_path, "r") as original_fp:
        original_content = original_fp.readlines()

    # Create a new XML file to write the modified content
    with open(new_sumo_net_file_path, "w") as new_fp:
        for line in original_content:
            modified_line = line
            # If the line contains "shape", modify it
            if "shape" in line:
                shape_index = line.index("shape")
                # Extract the shape value
                shape_value = line[shape_index:].split('"')[1]
                points = shape_value.split(" ")
                # Convert points to a list of tuples
                points = [tuple(map(float, point.split(","))) for point in points]
                new_points = []
                
                for point in points:
                    # Print the coordinates of each point
                    sumo_x, sumo_y, sumo_z = point
                    # Apply the UTM offset
                    utm_x = sumo_x - UTM_OFFSET[0]
                    utm_y = sumo_y - UTM_OFFSET[1]
                    utm_z = sumo_z - UTM_OFFSET[2]

                    lat, lon = utm.to_latlon(utm_x, utm_y, ZONE_NUMBER, ZONE_LETTER)
                    new_x, new_y = latlon_to_xy(GNSS_ORIGIN[0], GNSS_ORIGIN[1], lat, lon)
                    if new_x == sumo_x or new_y == sumo_y:
                        print()
                    carla_x = new_x
                    carla_y = -new_y
                    start_location = carla.Location(carla_x, carla_y, 300)
                    end_location = carla.Location(carla_x, carla_y, 200)
                    raycast_result = CARLA_WORLD.cast_ray(start_location, end_location)
                    if not raycast_result:
                        print("Ray did not hit the ground.")
                        carla_z = sumo_z - UTM_OFFSET[2]
                    else:
                        carla_z = min((item.location.z for item in raycast_result), default=sumo_z - UTM_OFFSET[2])

                    new_points.append(
                        [new_x, new_y, carla_z]
                    )

                # Convert the new points to the required format
                new_shape_value = " ".join(
                    [f"{point[0]},{point[1]},{point[2]}" for point in new_points]
                )
                   
                # Modify the line as needed
                modified_line = modified_line.replace(shape_value, new_shape_value)
            if " x=" in line and " y=" in line and " z=" in line:
                # Extract the coordinates
                x_index = line.index(" x=")
                y_index = line.index(" y=")
                z_index = line.index(" z=")
                # Extract the x, y, z values
                x_string = line[x_index:].split('"')[1]
                y_string = line[y_index:].split('"')[1]
                z_string = line[z_index:].split('"')[1]
                # Convert to float
                x_value = float(x_string)
                y_value = float(y_string)
                z_value = float(z_string)
                # Apply the UTM offset
                utm_x = x_value - UTM_OFFSET[0]
                utm_y = y_value - UTM_OFFSET[1]
                utm_z = z_value - UTM_OFFSET[2]
                lat, lon = utm.to_latlon(utm_x, utm_y, ZONE_NUMBER, ZONE_LETTER)
                new_x, new_y = latlon_to_xy(GNSS_ORIGIN[0], GNSS_ORIGIN[1], lat, lon)
                if new_x == x_value or new_y == y_value:
                    print()
                carla_x = new_x
                carla_y = -new_y
                start_location = carla.Location(carla_x, carla_y, 300)
                end_location = carla.Location(carla_x, carla_y, 200)
                raycast_result = CARLA_WORLD.cast_ray(start_location, end_location)
                if not raycast_result:
                    print("Ray did not hit the ground.")
                    carla_z = z_value - UTM_OFFSET[2]
                else:
                    carla_z = min((item.location.z for item in raycast_result), default=z_value - UTM_OFFSET[2])
                # Modify the line with the new coordinates
                modified_line = modified_line.replace(
                    f' x="{x_string}"', f' x="{new_x}"'
                ).replace(f' y="{y_string}"', f' y="{new_y}"').replace(
                    f' z="{z_string}"', f' z="{carla_z}"'
                )
                # Write the original line to the new file
            
            new_fp.write(modified_line)

    # Parse the XML file