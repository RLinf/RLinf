SFT for VLA / WAM Models
========================

Supervised fine-tuning (SFT) is the standard cold-start step before embodied RL: a strong SFT checkpoint dramatically reduces RL exploration time and improves final policy quality. This category lists RLinf's recipes for full-parameter and LoRA SFT on VLA / WAM models, plus VLM SFT for multimodal post-training.

After running SFT here, continue to :doc:`vla_wam_index` (model-centric RL) or :doc:`simulators_index` (benchmark-centric RL) to fine-tune the resulting checkpoint with RL.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <!-- TODO(thumbnail): replace placeholder cover image URL for sft_openpi -->
       <a href="embodied/sft_openpi.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_openpi.html" style="text-decoration: underline; color: blue;">
           <b>OpenPI Supervised Fine-Tuning</b>
         </a><br>
         Run full-parameter and LoRA SFT for OpenPI before RL fine-tuning
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sft_openpi_rlinf.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_openpi_rlinf.html" style="text-decoration: underline; color: blue;">
           <b>OpenPI Mixed-Precision SFT</b>
         </a><br>
         Run mixed-precision supervised fine-tuning with OpenPI_RLinf
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sft_dreamzero.html" style="display: block;"><img src="https://dreamzero0.github.io/images/project_overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_dreamzero.html" style="text-decoration: underline; color: blue;">
           <b>DreamZero Supervised Fine-Tuning</b>
         </a><br>
         Full-parameter and mixture SFT for DreamZero (WAN2.1 / WAN2.2 backbones)
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sft_fastwam.html" style="display: block;"><img src="https://yuantianyuan01.github.io/FastWAM/static/images/teaser_main.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_fastwam.html" style="text-decoration: underline; color: blue;">
           <b>FastWAM Evaluation and SFT</b>
         </a><br>
         Evaluate FastWAM on LIBERO and run full-parameter FSDP SFT
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sft_cosmos3.html" style="display: block;"><img src="https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/cosmos3-model-architecture.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_cosmos3.html" style="text-decoration: underline; color: blue;">
           <b>Cosmos3 Supervised Fine-Tuning</b>
         </a><br>
         LIBERO action-policy SFT on NVIDIA Cosmos3-Nano (OmniMoT)
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/sft_vlm.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_vlm.html" style="text-decoration: underline; color: blue;">
           <b>Qwen-VL</b>
         </a><br>
         Run full-parameter SFT and evaluation for VLM models such as Qwen
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dagger.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dagger.html" style="text-decoration: underline; color: blue;">
           <b>DAgger for Embodied Policies</b>
         </a><br>
         Guide online imitation learning with expert relabeling and replay-buffer updates
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   OpenPI <embodied/sft_openpi>
   OpenPI_RLinf <embodied/sft_openpi_rlinf>
   DreamZero <embodied/sft_dreamzero>
   FastWAM <embodied/sft_fastwam>
   Cosmos3 <embodied/sft_cosmos3>
   Qwen-VL <embodied/sft_vlm>
   DAgger <embodied/dagger>
