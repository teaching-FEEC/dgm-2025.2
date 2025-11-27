import torch

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.modules.generative.gan.fastgan.barlow_twins import BarlowTwinsProjector
from src.models.fastgan.fastgan import weights_init

# --- Path to checkpoint ---
ckpt_path = "logs/hypersynth/98xb02br/checkpoints/step=0-val_FID=80.2161.ckpt"
# ckpt_path = "logs/hypersynth/xhhsemy3/checkpoints/step=0-val_FID=173.2269.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

barlow_twins_projector_dim = 2048  # Set this to match your model's config
input_feature_dim = 512  # Set this to match your model's architecture

# Initialize the Barlow Twins projector
barlow_proj = BarlowTwinsProjector(
    input_dim=input_feature_dim,
    hidden_dim=barlow_twins_projector_dim,
    output_dim=barlow_twins_projector_dim,
)
barlow_proj.apply(weights_init)

# Attach the newly initialized projector weights to the checkpoint
ckpt["barlow_projector"] = barlow_proj.state_dict()

# --- Patch the discriminator optimizer group ---
optimizers_state = ckpt.get("optimizer_states", None)
if optimizers_state is not None and len(optimizers_state) > 1:
    d_opt_state = optimizers_state[1]  # Discriminator optimizer should be second
    # add barlow projector params to optimizer state
    param_group = d_opt_state["param_groups"][0]
    params = param_group["params"]
    # params is a list from 0 to N-1, we need to add new params for the barlow projector
    current_max_param_id = max(params)
    barlow_param_ids = list(range(current_max_param_id + 1, current_max_param_id + 1 + len(list(barlow_proj.parameters()))))
    params.extend(barlow_param_ids)
    param_group["params"] = params
    print("✅ Patched discriminator optimizer state in checkpoint.")
else:
    print("! Could not find discriminator optimizer state in checkpoint.")
    
# also remove hparams_name and hyper_parameters to avoid conflicts
if "hparams_name" in ckpt:
    del ckpt["hparams_name"]
if "hyper_parameters" in ckpt:
    del ckpt["hyper_parameters"]

# --- Save modified checkpoint ---
new_ckpt_path = ckpt_path.replace(".ckpt", "_barlow.ckpt")
torch.save(ckpt, new_ckpt_path)
print(f"✅ Patched checkpoint saved at:\n{new_ckpt_path}")