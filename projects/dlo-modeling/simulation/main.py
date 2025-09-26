import mujoco
import mujoco.viewer
import numpy as np
import time
import pyquaternion as pyq 
import glfw

def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements

def key_callback(key):
    if key == glfw.KEY_UP:  # Up arrow
        d.mocap_pos[0, 2] += 0.01
    elif key == 264:  # Down arrow
        d.mocap_pos[0, 2] -= 0.01
    elif key == 263:  # Left arrow
        d.mocap_pos[0, 0] -= 0.01
    elif key == 262:  # Right arrow
        d.mocap_pos[0, 0] += 0.01
    elif key == 320:  # Numpad 0
        d.mocap_pos[0, 1] += 0.01
    elif key == 330:  # Numpad .
        d.mocap_pos[0, 1] -= 0.01
    elif key == 260:  # Insert
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [1, 0, 0], 10)
    elif key == 261:  # Home
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [1, 0, 0], -10)
    elif key == 268:  # Home
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [0, 1, 0], 10)
    elif key == 269:  # End
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [0, 1, 0], -10)
    elif key == 266:  # Page Up
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [0, 0, 1], 10)
    elif key == 267:  # Page Down
        d.mocap_quat[0] = rotate_quaternion(d.mocap_quat[0], [0, 0, 1], -10)
    else:
        print(key)

# Load model and create data
#m = mujoco.MjModel.from_xml_path("./mujoco_menagerie/franka_emika_panda/panda.xml")

m = mujoco.MjModel.from_xml_path("model.xml")
d = mujoco.MjData(m)


print("Total bodies:", m.nbody)

    

# Get the ID of the body we want to track
body_id = m.body("link7").id

# Do forward kinematics
mujoco.mj_kinematics(m, d)

cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")


# Get the position of the body from the data
body_pos = d.xpos[body_id]
body_quat = d.xquat[body_id]
print(body_pos, body_quat)


with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as v:
    stepcount = 0
    #v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    #v.cam.fixedcamid = cam_id    
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        stepcount += 1
