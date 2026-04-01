import os
import torch
import shutil
from safetensors.torch import save_file

# ==========================================
# 📂 Path Configuration (ensure base_model_path is your original base model path)
# ==========================================
base_model_path = "/workspace/RLinf/GR00T-N1.6-3B"  # Original un-finetuned weights directory
sft_pt_path = "/workspace/RLinf/logs/20260323-14:22:31/gr00t_16_sft_libero/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt"
hf_output_path = "/workspace/RLinf/GR00T-1.6-SFT-LIBERO-HF" # New directory for the converted RL model

def main():
    os.makedirs(hf_output_path, exist_ok=True)

    # 1. Copy all non-weight files (config.json, experiment_cfg, and other metadata)
    print("📋 [1/3] Copying model configuration files...")
    for filename in os.listdir(base_model_path):
        # Exclude the old original weight files
        if not filename.endswith((".safetensors", ".pt", ".bin", ".index.json")):
            src = os.path.join(base_model_path, filename)
            dst = os.path.join(hf_output_path, filename)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # 2. Load the clean weights trained from SFT
    print("🔄 [2/3] Loading the SFT-trained PyTorch weights...")
    state_dict = torch.load(sft_pt_path, map_location="cpu")
    
    # Clean up redundant prefixes potentially left by FSDP
    clean_state_dict = {}
    for key, value in state_dict.items():
        # FSDP wrapping sometimes adds the '_fsdp_wrapped_module.' prefix, strip it off
        new_key = key.replace("_fsdp_wrapped_module.", "")
        clean_state_dict[new_key] = value

    # 3. Convert to safetensors (Loads faster and saves memory in the RL framework)
    print("💾 [3/3] Packaging into Hugging Face Safetensors format...")
    save_file(clean_state_dict, os.path.join(hf_output_path, "model.safetensors"))

    print(f"\n🎉 Conversion successfully completed!")
    print(f"👉 You can now point 'actor.model_path' in your RL YAML config to: {hf_output_path}")

if __name__ == "__main__":
    main()