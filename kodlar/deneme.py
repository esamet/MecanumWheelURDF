from math import radians
import math
import random
from isaacgym import gymapi
import numpy as np
gym = gymapi.acquire_gym()



sim_params = gymapi.SimParams()
compute_device_id=0
graphics_device_id=0
# get default set of parameters
sim_params = gymapi.SimParams()
# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -20.8)
# set PhysX-specific parameters
sim_params.physx.use_gpu = True 
sim_params.use_gpu_pipeline = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
# create sim with these parameters
physics_engine=gymapi.SIM_PHYSX
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)




# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0
# create the ground plane
gym.add_ground(sim, plane_params)



asset_root = "/home/samet/Desktop/robomaster-stl-files/wheels/kodlar/"
asset_file = "roboke.urdf"


asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.armature = 0.01
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# set up the env grid
num_envs = 12
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments

for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 0.0)

    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i,0 )
    actor_handles.append(actor_handle)

# configure the joints for effort control mode (once)
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    props["stiffness"].fill(0.0)
    props["damping"].fill(0.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

# apply efforts (every frame)
    #efforts = np.full(4, 1.0).astype(np.float32)
    FL_vel=3
    RL_vel=3
    FR_vel=-3
    RR_vel=-3
    efforts = np.array([-FL_vel, -FR_vel, RL_vel, RR_vel]).astype(np.float32)
    gym.apply_actor_dof_efforts(env, actor_handle, efforts)
    #joint_handle1 = gym.get_joint_handle(env, "MyActor", "RR_wheel_joint")
    #gym.set_joint_target_velocity(env, joint_handle1, float(1.0))


dof_names = gym.get_asset_dof_names(asset)
print(dof_names)
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
while not gym.query_viewer_has_closed(viewer):


    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
