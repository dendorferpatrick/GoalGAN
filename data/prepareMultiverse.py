import carla
import numpy as np

anchor_cameras = {
    "zara01": (carla.Transform(
        carla.Location(x=-33.863022, y=-56.820679, z=28.149984),
        carla.Rotation(pitch=-62.999184, yaw=-89.999214, roll=0.000053)), 30.0),
    "eth": (carla.Transform(
        carla.Location(x=42.360512, y=-29.728912, z=25.349985),
        carla.Rotation(pitch=-66.998413, yaw=-86.996346, roll=0.000057)), 85.0),
    "hotel": (carla.Transform(
        carla.Location(x=61.361576, y=-103.381432, z=22.765366),
        carla.Rotation(pitch=-63.188835, yaw=2.568912, roll=0.000136)), 45.0),
    "0000": (carla.Transform(
        carla.Location(x=-22.496830, y=-60.411972, z=12.070004),
        carla.Rotation(pitch=-30.999966, yaw=57.001354, roll=0.000025)), 65.0),
    "0400": (carla.Transform(
        carla.Location(x=-160.418839, y=33.280800, z=14.469944),
        carla.Rotation(pitch=-17.869482, yaw=54.943417, roll=0.000087)), 60.0),
    "0401": (carla.Transform(
        carla.Location(x=-120.234306, y=44.632133, z=10.556061),
        carla.Rotation(pitch=-26.056767, yaw=37.160381, roll=0.000123)), 70.0),
    "0500": (carla.Transform(
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600775, yaw=10.802525, roll=0.000002)), 30.0),
}
anchor_cameras["zara02"] = anchor_cameras["zara01"]

recording_cameras = {
    "zara01": [(carla.Transform(  # anchor-view
        carla.Location(x=-33.863022, y=-56.820679, z=28.149984),
        carla.Rotation(pitch=-62.999184, yaw=-89.999214, roll=0.000053)), 55.0),
               (carla.Transform(  # left
        carla.Location(x=-58.868965, y=-44.742245, z=28.149984),
        carla.Rotation(pitch=-37.998379, yaw=-49.998241, roll=0.000101)), 35.0),
               (carla.Transform(  # right
        carla.Location(x=-19.291773, y=-50.881275, z=28.149984),
        carla.Rotation(pitch=-45.998478, yaw=-118.998390, roll=0.00012)), 45.0),
               (carla.Transform(  # top-down
        carla.Location(x=-34.080410, y=-70.821098, z=54.799969),
        carla.Rotation(pitch=-84.997345, yaw=-88.997765, roll=0.000078)), 35.0),
    ],
    "eth": [(carla.Transform(  # anchor-view
        carla.Location(x=41.742561, y=-11.228123, z=22.499983),
        carla.Rotation(pitch=-38.997192, yaw=-89.996033, roll=0.000101)), 80.0),
               (carla.Transform(  # left
        carla.Location(x=13.706966, y=-17.478188, z=32.549980),
        carla.Rotation(pitch=-40.996883, yaw=-32.996113, roll=0.000096)), 55.0),
               (carla.Transform(  # right
        carla.Location(x=71.222115, y=-21.480639, z=32.499981),
        carla.Rotation(pitch=-39.996334, yaw=-156.996201, roll=0.000133)), 50.0),
               (carla.Transform(  # top-down
        carla.Location(x=40.933189, y=-24.631042, z=91.099937),
        carla.Rotation(pitch=-79.995010, yaw=-88.995430, roll=0.000216)), 50.0),
    ],
    "hotel": [(carla.Transform(  # anchor-view
        carla.Location(x=58.757435, y=-101.250473, z=25.415363),
        carla.Rotation(pitch=-64.188843, yaw=2.568922, roll=0.000135)), 65.0),
               (carla.Transform(  # left
        carla.Location(x=62.436810, y=-117.175545, z=19.665363),
        carla.Rotation(pitch=-40.187798, yaw=75.567665, roll=0.000085)), 65.0),
               (carla.Transform(  # right
        carla.Location(x=58.515789, y=-86.332535, z=19.665363),
        carla.Rotation(pitch=-38.187317, yaw=-59.432423, roll=0.000065)), 65.0),
               (carla.Transform(  # top-down
        carla.Location(x=66.663460, y=-102.476425, z=30.865358),
        carla.Rotation(pitch=-88.958252, yaw=-179.104660, roll=-179.895248)), 50.0),
    ],
    "0000": [(carla.Transform(  # anchor-view
        carla.Location(x=-21.545109, y=-61.469452, z=12.120005),
        carla.Rotation(pitch=-39.999821, yaw=65.000923, roll=0.000018)), 90.0),
               (carla.Transform(  # left
        carla.Location(x=-7.899245, y=-64.047493, z=12.120005),
        carla.Rotation(pitch=-37.999504, yaw=106.000496, roll=0.000020)), 90.0),
               (carla.Transform(  # right
        carla.Location(x=-38.025734, y=-52.780418, z=11.870004),
        carla.Rotation(pitch=-30.999201, yaw=32.000214, roll=0.000047)), 70.0),
               (carla.Transform(  # top-down
        carla.Location(x=-12.978075, y=-32.058861, z=48.219952),
        carla.Rotation(pitch=-87.999031, yaw=0.000000, roll=0.000000)), 70.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-20.141300, y=-55.388958, z=2.220005),
        #carla.Rotation(pitch=0.000260, yaw=66.998703, roll=0.000035)), 90.0),
    ],
    "0400": [(carla.Transform(  # anchor-view
        carla.Location(x=-163.437454, y=26.059809, z=20.669943),
        carla.Rotation(pitch=-19.869471, yaw=36.942986, roll=0.000073)), 60.0),
               (carla.Transform(  # left
        carla.Location(x=-114.143768, y=7.405876, z=23.469940),
        carla.Rotation(pitch=-27.869444, yaw=81.942535, roll=0.000055)), 60.0),
               (carla.Transform(  # right
        carla.Location(x=-173.366577, y=70.659279, z=23.469940),
        carla.Rotation(pitch=-23.869440, yaw=-1.057342, roll=0.000046)), 55.0),
               (carla.Transform(  # top-down
        carla.Location(x=-107.249977, y=49.348232, z=101.969933),
        carla.Rotation(pitch=-83.868240, yaw=89.941933, roll=0.000096)), 55.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-163.437454, y=26.059809, z=0.819942),
        #carla.Rotation(pitch=5.130530, yaw=42.942547, roll=0.000073)), 60.0),
    ],
    "0401": [(carla.Transform(  # anchor-view
        carla.Location(x=-128.780029, y=31.252804, z=16.156065),
        carla.Rotation(pitch=-26.056767, yaw=42.160397, roll=0.000124)), 80.0),
               (carla.Transform(  # left
        carla.Location(x=-101.373863, y=15.802762, z=16.156065),
        carla.Rotation(pitch=-26.056761, yaw=91.160004, roll=0.000150)), 75.0),
               (carla.Transform(  # right
        carla.Location(x=-139.725403, y=61.328167, z=16.156065),
        carla.Rotation(pitch=-30.818111, yaw=-1.363098, roll=0.000145)), 80.0),
               (carla.Transform(  # top-down
        carla.Location(x=-109.142944, y=58.624207, z=70.706039),
        carla.Rotation(pitch=-80.815720, yaw=0.636051, roll=0.000164)), 65.0),
        #       (carla.Transform(  # dashboard view
        #carla.Location(x=-128.780029, y=31.252802, z=1.306065),
        #carla.Rotation(pitch=3.943252, yaw=42.161617, roll=0.000125)), 80.0),
    ],
    "0500": [(carla.Transform(  # anchor-view
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=10.802557, roll=0.000002)), 35.0),
               (carla.Transform(  # left
        carla.Location(x=-150.165619, y=-129.959244, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=42.802635, roll=0.000008)), 35.0),
               (carla.Transform(  # right
        carla.Location(x=-157.999283, y=-55.524170, z=26.075722),
        carla.Rotation(pitch=-23.600767, yaw=-33.197342, roll=0.000007)), 35.0),
               (carla.Transform(  # top-down
        carla.Location(x=-100.003044, y=-96.517174, z=52.925720),
        carla.Rotation(pitch=-78.599899, yaw=89.801888, roll=0.000000)), 70.0),
        #       (carla.Transform( # dashboard view
        #carla.Location(x=-144.576553, y=-97.665466, z=1.325722),
        #carla.Rotation(pitch=2.399254, yaw=16.803694, roll=0.000003)), 45.0),
    ],
}
recording_cameras["zara02"] = recording_cameras["zara01"]


anchor_cameras_annotation = {
    "zara01": (carla.Transform(
        carla.Location(x=-33.937153, y=-65.975639, z=13.199974),
        carla.Rotation(pitch=-63.998699, yaw=-90.999649, roll=0.000117)), 90.0),
    "eth": (carla.Transform(
        carla.Location(x=41.688187, y=-16.916178, z=25.349985),
        carla.Rotation(pitch=-44.997559, yaw=-86.996063, roll=0.000133)), 90.0),
    "hotel": (carla.Transform(
        carla.Location(x=62.348896, y=-101.509659, z=22.765366),
        carla.Rotation(pitch=-69.188515, yaw=-0.431061, roll=0.000136)), 90.0),
    "0000": (carla.Transform(
        carla.Location(x=-21.634167, y=-60.972176, z=12.070004),
        carla.Rotation(pitch=-30.999966, yaw=59.001438, roll=0.000028)), 90.0),
    "0400": (carla.Transform(
        carla.Location(x=-160.418839, y=33.280800, z=14.469944),
        carla.Rotation(pitch=-17.869482, yaw=54.943417, roll=0.000087)), 90.0),
    "0401": (carla.Transform(
        carla.Location(x=-120.234306, y=44.632133, z=10.556061),
        carla.Rotation(pitch=-26.056767, yaw=37.160381, roll=0.000123)), 90.0),
    "0500": (carla.Transform(
        carla.Location(x=-154.292542, y=-99.519508, z=26.075722),
        carla.Rotation(pitch=-23.600775, yaw=10.802525, roll=0.000002)), 90.0),
}
anchor_cameras_annotation["zara02"] = anchor_cameras_annotation["zara01"]



def compute_extrinsic_from_transform(transform_):
  """
  Creates extrinsic matrix from carla transform.
  This is known as the coordinate system transformation matrix.
  """

  rotation = transform_.rotation
  location = transform_.location
  c_y = np.cos(np.radians(rotation.yaw))
  s_y = np.sin(np.radians(rotation.yaw))
  c_r = np.cos(np.radians(rotation.roll))
  s_r = np.sin(np.radians(rotation.roll))
  c_p = np.cos(np.radians(rotation.pitch))
  s_p = np.sin(np.radians(rotation.pitch))
  matrix = np.matrix(np.identity(4))  # matrix is needed
  # 3x1 translation vector
  matrix[0, 3] = location.x
  matrix[1, 3] = location.y
  matrix[2, 3] = location.z
  # 3x3 rotation matrix
  matrix[0, 0] = c_p * c_y
  matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
  matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
  matrix[1, 0] = s_y * c_p
  matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
  matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
  matrix[2, 0] = s_p
  matrix[2, 1] = -c_p * s_r
  matrix[2, 2] = c_p * c_r
  # [3, 3] == 1, rest is zero
  return matrix


# # transform to the world origin
# camera_rt = compute_extrinsic_from_transform(
#     args_.rgb_camera.camera_actor.get_transform())
# click_point_world_3d = np.dot(camera_rt, click_point_3d)
# x, y, z = click_point_world_3d.tolist()[0][:3]  # since it is np.matrix
# xyz = [x, y, z + 0.1]  # slightly above ground


if __name__=="__main__":
    width = 1920.0
    homographies = {}
    for d, scene in recording_cameras.items():

        top_down = scene[-1]

        carla, fov = top_down
        z = carla.location.z
        print( 2 * np.tanh(fov * np.pi/360.)* z)
        homographies[d] = z * 2 * np.tanh(fov * np.pi/360.)/ width

    print(homographies)



    #
    # matrix = compute_extrinsic_from_transform(recording_cameras["zara01"][-1][0])
    #
    # point = np.array([0, 0, 0,  1]).reshape(4, 1)
    #
    #
    # point_xy = np.dot(matrix, point)[:3]
    # print( point_xy)
    # print(recording_cameras["zara01"][-1][0].location.x)
    # print(recording_cameras["zara01"][-1][0].location.y)
    # print(recording_cameras["zara01"][-1][0].location.z)
    #
    # point = np.array([100, 0, 0, 1]).reshape(4, 1)
    # point_xy = np.dot(matrix, point)[:3]
    # print( point_xy)
    # print(recording_cameras["zara01"][-1][0].location.x)
    # print(recording_cameras["zara01"][-1][0].location.y)
    # print(recording_cameras["zara01"][-1][0].location.z)