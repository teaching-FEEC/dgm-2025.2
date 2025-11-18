import numpy as np
import mujoco
# import mujoco.viewer
import re
import time
import concurrent.futures
import os
import multiprocessing 

# --- Simulation Parameters ---
XML_PATH = "rope_chain.xml"
MIN_FORCE_MAG = 0.1  # Minimum force magnitude
MAX_FORCE_MAG = 3.0  # Maximum force magnitude
FORCE_STEPS = 200
NUM_TRANSITIONS = 10000  # Total number of transitions to collect (across all workers)
NUM_ROLLOUTS = 5       # Number of parallel workers/rollouts
SETTLE_TIME = 5.0
SAVE_INTERVAL = 10   # <-- NEU: Print progress every X steps (per worker)
USE_VIEWER = False       # MUST be False for multiprocessing

# --- Dateinamen ---
FINAL_OUTPUT_FILENAME = "rope_state_action_next_state.npz"
PARTIAL_FILENAME_TPL = "rope_data_part_{worker_id}.npz"


def get_link_ids(model, prefix="link_"):
    """
    Finds the body IDs for all bodies with a given prefix.
    This is necessary because MuJoCo shuffles body IDs.
    """
    ids, names = [], []
    i = 0
    while True:
        name = f"{prefix}{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid == -1:
            break
        ids.append(bid)
        names.append(name)
        i += 1
        if i > model.nbody:
            break
    if ids:
        return np.array(ids, dtype=int), names
        
    # Fallback scan if names aren't contiguous (e.g., link_0, link_2)
    cand = []
    for b in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", nm)
        if m:
            cand.append((int(m.group(1)), b, nm))
    if not cand:
        raise RuntimeError(f"No bodies named like '{prefix}*' found.")
    cand.sort(key=lambda x: x[0])
    return np.array([b for _, b, _ in cand], dtype=int), [nm for _, _, nm in cand]

def save_data(filename, num_to_save, states, actions, next_states, link_names_arr, meta):
    """Saves a slice of the collected data to an .npz file."""
    print(f"\n--- Saving {num_to_save} transitions to {filename} ---")
    np.savez_compressed(
        filename,
        states=states[:num_to_save],
        actions=actions[:num_to_save],
        next_states=next_states[:num_to_save],
        link_names=link_names_arr,
        meta=meta,
    )
    print(f"  states:      {states[:num_to_save].shape}")
    print(f"  actions:     {actions[:num_to_save].shape}")
    print(f"  next_states: {next_states[:num_to_save].shape}")
    print("--- Save complete ---")


def run_rollout_worker(worker_id, transitions_for_this_worker, meta_info, 
                     link_names_arr, print_lock): # <-- NEU: print_lock
    """
    This function is executed by each parallel worker.
    It runs a simulation segment and saves its results to a partial file.
    """
    
    # Each worker must load its own model and data
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    link_ids, _ = get_link_ids(m)
    L = len(link_ids)

    # Initialize local data arrays
    states = np.zeros((transitions_for_this_worker, L, 3), dtype=np.float32)
    actions = np.zeros((transitions_for_this_worker, 4), dtype=np.float32)
    next_states = np.zeros((transitions_for_this_worker, L, 3), dtype=np.float32)

    # Each worker should have a different random seed
    seed = (os.getpid() * int(time.time() * 1000) + worker_id) % (2**32)
    np.random.seed(seed)

    def step():
        mujoco.mj_step(m, d)

    # Settle simulation
    for _ in range(int(SETTLE_TIME / m.opt.timestep)):
        step()

    # --- Run simulation loop ---
    for i in range(transitions_for_this_worker):
        # 1. Record current state (S_t)
        states[i] = d.xpos[link_ids, :].astype(np.float32)

        # 2. Choose and record action (A_t)
        li = np.random.randint(0, L)
        bid = link_ids[li]
        
        fvec = np.zeros(3, dtype=np.float32)
        axis_idx = np.random.randint(0, 3)
        sign = 1.0 if np.random.rand() < 0.5 else -1.0
        magnitude = np.random.uniform(MIN_FORCE_MAG, MAX_FORCE_MAG)
        fvec[axis_idx] = sign * magnitude

        actions[i, :3] = fvec
        actions[i,  3] = float(li)
        
        # 3. Apply action
        for _ in range(FORCE_STEPS):
            d.xfrc_applied[bid, :3] = fvec
            step()
        d.xfrc_applied[bid, :] = 0.0

        # 4. Record the resulting next state (S_{t+1})
        next_states[i] = d.xpos[link_ids, :].astype(np.float32)

        # --- (NEU) Progress Logging (mit Lock) ---
        if (i + 1) % SAVE_INTERVAL == 0:
            with print_lock:
                print(f"[Worker {worker_id}] Progress: {i + 1}/{transitions_for_this_worker} transitions completed.")

    # --- Save partial data ---
    output_filename = PARTIAL_FILENAME_TPL.format(worker_id=worker_id)
    # Verwende das print_lock auch hier, um die "Saving..."-Nachricht zu schützen
    with print_lock:
        save_data(output_filename, transitions_for_this_worker, states, actions, 
                  next_states, link_names_arr, meta_info)
    
    return output_filename, transitions_for_this_worker


def main():
    """
    Main function to ORCHESTRATE the simulation.
    It distributes work to parallel processes and concatenates the results.
    """
    if USE_VIEWER:
        print("ERROR: USE_VIEWER must be set to False to run in parallel.")
        print("Parallel workers cannot (and should not) launch GUI viewers.")
        return

    # --- Load model once in main process to get metadata ---
    try:
        m_main = mujoco.MjModel.from_xml_path(XML_PATH)
        _, link_names = get_link_ids(m_main)
        link_names_arr = np.array(link_names, dtype=object)
    except Exception as e:
        print(f"Error loading XML {XML_PATH}: {e}")
        return

    meta_info = dict(
        xml=XML_PATH,
        timestep=m_main.opt.timestep,
        min_force_mag=MIN_FORCE_MAG,
        max_force_mag=MAX_FORCE_MAG,
        force_steps=FORCE_STEPS,
        settle_time=SETTLE_TIME,
        num_rollouts=NUM_ROLLOUTS,
    )
    
    # --- Distribute work ---
    base_transitions = NUM_TRANSITIONS // NUM_ROLLOUTS
    remainder = NUM_TRANSITIONS % NUM_ROLLOUTS
    transitions_per_worker = [base_transitions] * NUM_ROLLOUTS
    for i in range(remainder):
        transitions_per_worker[i] += 1
        
    print(f"Starting {NUM_ROLLOUTS} parallel workers...")
    print(f"Total transitions to collect: {NUM_TRANSITIONS}")
    print(f"Work distribution: {transitions_per_worker}")
    
    global_start_time = time.perf_counter()
    partial_files = []
    
    # --- (NEU) Manager erstellen, um Lock zu teilen ---
    with multiprocessing.Manager() as manager:
        print_lock = manager.Lock() # Lock für saubere Konsolenausgabe
        
        # --- Launch parallel workers ---
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_ROLLOUTS) as executor:
            futures = []
            for i in range(NUM_ROLLOUTS):
                n_tasks = transitions_per_worker[i]
                if n_tasks > 0:
                    print(f"  Submitting worker {i} for {n_tasks} transitions.")
                    futures.append(
                        executor.submit(
                            run_rollout_worker, 
                            i, 
                            n_tasks, 
                            meta_info, 
                            link_names_arr,
                            print_lock  # <-- Lock an Worker übergeben
                        )
                    )

            # Wait for results
            for future in concurrent.futures.as_completed(futures):
                try:
                    filename, num_done = future.result()
                    # Diese Nachricht ist jetzt redundant, da save_data() sie druckt
                    # print(f"  Worker finished, saved {num_done} transitions to {filename}")
                    partial_files.append(filename)
                except Exception as e:
                    print(f"A worker failed: {e}")

    total_time = time.perf_counter() - global_start_time
    print(f"\nAll workers finished in {total_time:.2f}s.")

    if not partial_files:
        print("No data was collected. Exiting.")
        return

    # --- Concatenate results ---
    print(f"Concatenating {len(partial_files)} partial files...")
    all_states, all_actions, all_next_states = [], [], []
    total_saved_transitions = 0

    partial_files.sort() 

    for filename in partial_files:
        try:
            with np.load(filename) as data:
                all_states.append(data['states'])
                all_actions.append(data['actions'])
                all_next_states.append(data['next_states'])
                total_saved_transitions += len(data['states'])
        except Exception as e:
            print(f"Error loading {filename}: {e}. Skipping...")
        finally:
            # Clean up partial file
            if os.path.exists(filename):
                os.remove(filename)
                print(f"  Removed temporary file: {filename}")

    if total_saved_transitions == 0:
        print("Concatenation failed or no data was loaded. Final file not saved.")
        return

    # Combine all data
    final_states = np.concatenate(all_states, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    final_next_states = np.concatenate(all_next_states, axis=0)

    # --- Final Save ---
    print("\nPerforming final save...")
    save_data(FINAL_OUTPUT_FILENAME, total_saved_transitions, final_states, 
              final_actions, final_next_states, link_names_arr, meta_info)
    
    # --- Final Stats ---
    if total_saved_transitions > 0:
        avg_time_per_transition = total_time / total_saved_transitions
        transitions_per_sec = total_saved_transitions / total_time
        print(f"  Final Stats ({total_saved_transitions} transitions):")
        print(f"    Total Runtime: {total_time:.2f}s")
        print(f"    Avg. Time/Transition: {avg_time_per_transition * 1000:.4f} ms")
        print(f"    Throughput: {transitions_per_sec:.2f} trans/sec")
    else:
        print(f"  Final Stats: No transitions recorded.")
        print(f"    Total Runtime: {total_time:.2f}s")
    print("=================================================")


if __name__ == "__main__":
    main()