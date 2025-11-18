# collect_states_actions.py
import numpy as np
import mujoco
import mujoco.viewer
import re
import time  # <--- Added this import

XML_PATH = "rope_chain.xml"
FORCE_MAG = 2.0
FORCE_STEPS = 100
NUM_TRANSITIONS = 100000
SETTLE_TIME = 5.0
USE_VIEWER = True           # set True if you want to watch

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
    # Fallback scan (if names aren’t contiguous)
    cand = []
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", nm)
        if m: cand.append((int(m.group(1)), b, nm))
    if not cand:
        raise RuntimeError("No bodies named link_0, link_1, ... found.")
    cand.sort(key=lambda x: x[0])
    return np.array([b for _, b, _ in cand], dtype=int), [nm for _, _, nm in cand]

def sample_force(mag):
    axis = np.zeros(3); axis[np.random.randint(0, 3)] = np.random.uniform(0.5, 2.0)
    if np.random.rand() < 0.5: axis = -axis
    return mag * axis

def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    link_ids, link_names = get_link_ids(m)
    L = len(link_ids)

    # Outputs you asked for
    states  = np.zeros((NUM_TRANSITIONS, L, 3), dtype=np.float32)
    actions = np.zeros((NUM_TRANSITIONS, 4), dtype=np.float32)  # [dx,dy,dz, link_id_as_float]
    link_index = np.zeros((NUM_TRANSITIONS,), dtype=np.int32)   # same info, but as int

    def step():
        mujoco.mj_step(m, d)

    # Settle first
    for _ in range(int(SETTLE_TIME / m.opt.timestep)):
        step()

    # <--- Added global start time
    global_start_time = time.perf_counter()

    if USE_VIEWER:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            for i in range(NUM_TRANSITIONS):
                # <--- Added transition start time
                transition_start_time = time.perf_counter()
                
                # print(i) # <--- Replaced this
                s_before = np.array(d.xpos[link_ids, :], dtype=np.float64)
                states[i] = s_before

                li = np.random.randint(0, L)
                bid = link_ids[li]
                fvec = sample_force(FORCE_MAG)

                for _ in range(FORCE_STEPS):
                    d.xfrc_applied[bid, :3] = fvec
                    step(); viewer.sync(); 
                d.xfrc_applied[bid, :] = 0.0
                time.sleep(0.5) 
                s_after = np.array(d.xpos[link_ids, :], dtype=np.float64)
                offset = s_after[li] - s_before[li]

                actions[i, :3] = offset.astype(np.float32)
                actions[i,  3] = float(li)
                link_index[i]  = li

                # <--- Added timing calculations and print
                current_time = time.perf_counter()
                transition_time = current_time - transition_start_time
                total_time = current_time - global_start_time
                print(f"Transition {i+1}/{NUM_TRANSITIONS} | "
                      f"Transition Time: {transition_time:.4f}s | "
                      f"Total Runtime: {total_time:.2f}s")
    else:
        for i in range(NUM_TRANSITIONS):
            # <--- Added transition start time
            transition_start_time = time.perf_counter()
            
            # print(i) # <--- Replaced this
            s_before = np.array(d.xpos[link_ids, :], dtype=np.float64)
            states[i] = s_before

            li = np.random.randint(0, L)
            bid = link_ids[li]
            fvec = sample_force(FORCE_MAG)

            for _ in range(FORCE_STEPS):
                d.xfrc_applied[bid, :3] = fvec
                step()
            d.xfrc_applied[bid, :] = 0.0

            s_after = np.array(d.xpos[link_ids, :], dtype=np.float64)
            offset = s_after[li] - s_before[li]

            actions[i, :3] = offset.astype(np.float32)
            actions[i,  3] = float(li)
            link_index[i]  = li

            # <--- Added timing calculations and print
            current_time = time.perf_counter()
            transition_time = current_time - transition_start_time
            total_time = current_time - global_start_time
            print(f"Transition {i+1}/{NUM_TRANSITIONS} | "
                  f"Transition Time: {transition_time:.4f}s | "
                  f"Total Runtime: {total_time:.2f}s")

    np.savez_compressed(
        "100000_rope_states_actions_2.npz",
        states=states,                 # (N, L, 3)
        actions=actions,               # (N, 4) -> [dx,dy,dz, link_id_as_float]
        link_index=link_index,         # (N,)   -> same link id, as int
        link_names=np.array(link_names, dtype=object),
        meta=dict(
            xml=XML_PATH,
            timestep=m.opt.timestep,
            force_mag=FORCE_MAG,
            force_steps=FORCE_STEPS,
            settle_time=SETTLE_TIME,
        ),
    )
    print("Saved rope_states_actions.npz")
    print("states:", states.shape, "actions:", actions.shape)

if __name__ == "__main__":
    main()