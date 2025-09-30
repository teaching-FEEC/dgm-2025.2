# run with mjpython to get viewer; if no viewer is needed, set USE_VIEWER to false
import numpy as np
import mujoco
import mujoco.viewer
import re

XML_PATH = "rope_chain.xml"
FORCE_MAG = 2.0
FORCE_STEPS = 100
NUM_TRANSITIONS = 500
SETTLE_TIME = 5.0        # seconds to settle before collecting
USE_VIEWER = True

def get_link_ids(model, prefix="link_"):
    ids, names = [], []
    i = 0
    while True:
        name = f"{prefix}{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1: break
        ids.append(bid); names.append(name); i += 1
        if i > model.nbody: break
    if ids:
        return np.array(ids, dtype=int), names

    # Fallback scan
    cand = []
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", nm)
        if m:
            cand.append((int(m.group(1)), b, nm))
    if not cand:
        raise RuntimeError("No link_* bodies found.")
    cand.sort(key=lambda x: x[0])
    ids = np.array([b for _, b, _ in cand], dtype=int)
    names = [nm for _, _, nm in cand]
    return ids, names

def sample_force(mag):
    axis = np.zeros(3); axis[np.random.randint(0, 3)] = 1.0
    if np.random.rand() < 0.5: axis = -axis
    return mag * axis

def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    link_ids, link_names = get_link_ids(m)
    L = len(link_ids)

    samples = np.zeros((NUM_TRANSITIONS, L, 4), dtype=np.float32)

    def step():
        mujoco.mj_step(m, d)

    # settle
    settle_steps = int(SETTLE_TIME / m.opt.timestep)
    if USE_VIEWER:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            for _ in range(settle_steps):
                step(); viewer.sync()

            for i in range(NUM_TRANSITIONS):
                # state before burst
                print(i)
                s = d.xpos[link_ids, :]           # (L,3)
                samples[i, :, 0:3] = s
                # choose action
                li = np.random.randint(0, L)
                a = sample_force(FORCE_MAG)
                bid = link_ids[li]
                samples[i, :, 3] = li             # broadcast link id

                # apply force for FORCE_STEPS
                for _ in range(FORCE_STEPS):
                    d.xfrc_applied[bid, :3] = a
                    step(); viewer.sync()
                d.xfrc_applied[bid, :] = 0.0
    else:
        for _ in range(settle_steps):
            step()
        for i in range(NUM_TRANSITIONS):
            print(i)
            s = d.xpos[link_ids, :]
            samples[i, :, 0:3] = s
            li = np.random.randint(0, L)
            a = sample_force(FORCE_MAG)
            bid = link_ids[li]
            samples[i, :, 3] = li
            for _ in range(FORCE_STEPS):
                d.xfrc_applied[bid, :3] = a
                step()
            d.xfrc_applied[bid, :] = 0.0

    # save single matrix
    np.savez_compressed(
        "rope_minimal.npz",
        samples=samples,             # (N, L, 4): xyz + acted_link_id
        link_names=np.array(link_names, dtype=object),
        meta=dict(
            xml=XML_PATH,
            timestep=m.opt.timestep,
            force_mag=FORCE_MAG,
            force_steps=FORCE_STEPS,
            settle_time=SETTLE_TIME,
        ),
    )
    print("Saved rope_minimal.npz with samples shape", samples.shape)

if __name__ == "__main__":
    main()