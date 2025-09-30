import numpy as np
d = np.load("rope_minimal.npz", allow_pickle=True)
S = d["samples"]  # (N, L, 4)
print(S.shape)

# Example: first transition
xyz = S[0, :, :3]      # (L,3)
acted_link = int(S[0, 0, 3])   # same for all links in transition 0
print("acted_link:", acted_link)
print(xyz)
