import time
import numpy as np
import mujoco
import mujoco.viewer

# -------------------- Tunables --------------------
integration_dt: float = 0.1
damping: float = 1e-4
Kpos: float = 0.95
Kori: float = 0.95
gravity_compensation: bool = True
dt: float = 0.002
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
max_angvel = 0.785
# --------------------------------------------------

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Upgrade to mujoco 3.1.0 or later."

    # Load model/data
    model = mujoco.MjModel.from_xml_path("scene_with_hand.xml")
    data = mujoco.MjData(model)

    # ---- Viewer key callback + gripper control ----
    grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    state = {"open": False}

    def set_gripper(open_: bool):
        lo, hi = model.actuator_ctrlrange[grip_act]
        data.ctrl[grip_act] = hi if open_ else lo  # 255=open, 0=close per your XML

    def on_key(keycode: int):
        # int keycode; avoid glfw dependency
        if keycode == ord(' '):        # SPACE toggles
            state["open"] = not state["open"]
        elif keycode in (ord('o'), ord('O')):
            state["open"] = True
        elif keycode in (ord('c'), ord('C')):
            state["open"] = False

    # ---- Model options ----
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # ---- Named handles ----
    site_id = model.site("attachment_site").id         # end-effector site
    key_id  = model.key("home").id                     # initial keyframe

    # Arm joint names (position actuators exist for these)
    arm_joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
    arm_joint_ids   = np.array([model.joint(n).id for n in arm_joint_names], dtype=int)
    arm_act_ids     = np.array([model.actuator(n).id for n in arm_joint_names], dtype=int)

    # ---------------- Address maps (scales to 200+ joints) ----------------
    jtype    = np.asarray(model.jnt_type, dtype=int)
    qposadr  = np.asarray(model.jnt_qposadr, dtype=int)
    dofadr   = np.asarray(model.jnt_dofadr, dtype=int)

    HINGE = mujoco.mjtJoint.mjJNT_HINGE
    SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
    BALL  = mujoco.mjtJoint.mjJNT_BALL
    FREE  = mujoco.mjtJoint.mjJNT_FREE

    is_scalar = (jtype == HINGE) | (jtype == SLIDE)      # only these have scalar limits
    scalar_qpos_idx = qposadr[is_scalar]                 # indices into qpos to clamp
    scalar_lo = model.jnt_range[is_scalar, 0]
    scalar_hi = model.jnt_range[is_scalar, 1]

    # Arm indices into qpos (positions) and qvel/DOF (Jacobian columns)
    arm_qpos_idx = qposadr[arm_joint_ids]                # length 7
    arm_qvel_idx = dofadr[arm_joint_ids]                 # length 7
    assert arm_qpos_idx.size == 7 and arm_qvel_idx.size == 7

    # Home pose for arm (from keyframe)
    q0_arm = model.key("home").qpos[arm_qpos_idx]

    # Preallocations
    jac = np.zeros((6, model.nv))                        # site Jacobian (6 x nv)
    diag = damping * np.eye(6)
    eye7 = np.eye(7)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # ---------------- Viewer ----------------
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=on_key,
    ) as viewer:
        # Reset to keyframe and camera
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE



        while viewer.is_running():
            step_start = time.time()

            for i in range(model.nbody):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name is None:
                    name = f"body_{i}"   # fallback name
                pos = data.xpos[i]
                #print(f"{i:3d} {name:20s} {pos}")
                            
            # ---------- Task-space twist (6D) ----------
            # Position error
            dx = data.mocap_pos[model.body("target").mocapid[0]] - data.site(site_id).xpos
            twist[:3] = Kpos * dx / integration_dt

            # Orientation error: quat(target) * conj(quat(site)) → axis-angle velocity
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[model.body("target").mocapid[0]], site_quat_conj)
            mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
            twist[3:] *= Kori / integration_dt

            # ---------- Jacobian (6 x nv) at site ----------
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            J  = jac[:, arm_qvel_idx]                     # 6 x 7
            JT = J.T

            # ---------- DLS IK for arm qvel (7,) ----------
            dq_arm = JT @ np.linalg.solve(J @ JT + diag, twist)

            # Nullspace to home
            q_arm = data.qpos[arm_qpos_idx]               # (7,)
            Jpinv = JT @ np.linalg.inv(J @ JT + diag)     # 7 x 6
            dq_arm += (eye7 - Jpinv @ J) @ (Kn * (q0_arm - q_arm))

            # Speed clamp
            m = np.abs(dq_arm).max()
            if m > max_angvel:
                dq_arm *= max_angvel / m

            # ---------- Integrate full qpos ----------
            dq_full = np.zeros(model.nv)
            dq_full[arm_qvel_idx] = dq_arm

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq_full, integration_dt)

            # Clamp ONLY scalar joints; never touch ball/free pieces
            if scalar_qpos_idx.size:
                q[scalar_qpos_idx] = np.clip(q[scalar_qpos_idx], scalar_lo, scalar_hi)

            # Keep quats valid, then write back & sync kinematics
            mujoco.mj_normalizeQuat(model, q)
            data.qpos[:] = q
            mujoco.mj_fwdPosition(model, data)


            # Gripper: force/general control via single actuator
            set_gripper(state["open"])

            # ---------- Step dynamics & draw ----------
            mujoco.mj_step(model, data)
            viewer.sync()

            # Soft real-time
            dt_left = dt - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

if __name__ == "__main__":
    main()