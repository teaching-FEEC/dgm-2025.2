import numpy as np
d = np.load("rope_states_actions.npz", allow_pickle=True)
S = d["states"]    # (N, L, 3)
A = d["actions"]   # (N, 4)  [dx,dy,dz, link_id_as_float]
li = d["link_index"]

print(S.shape, A.shape)
print("First action offset:", A[0, :3], "link:", int(A[0, 3]))
print(S)