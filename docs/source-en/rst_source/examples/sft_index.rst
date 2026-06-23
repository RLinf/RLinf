SFT for VLA / WAM Models
========================

Supervised fine-tuning (SFT) is the standard cold-start step before embodied RL: a strong SFT checkpoint dramatically reduces RL exploration time and improves final policy quality. This category lists RLinf's recipes for full-parameter and LoRA SFT on VLA / WAM models, plus VLM SFT for multimodal post-training.

After running SFT here, continue to :doc:`vla_wam_index` (model-centric RL) or :doc:`simulators_index` (benchmark-centric RL) to fine-tune the resulting checkpoint with RL.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <!-- TODO(thumbnail): replace placeholder cover image URL for sft_openpi -->
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_openpi.html" style="text-decoration: underline; color: blue;">
           <b>OpenPI Supervised Fine-Tuning</b>
         </a><br>
         Run full-parameter and LoRA SFT for OpenPI before RL fine-tuning
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://dreamzero0.github.io/images/project_overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_dreamzero.html" style="text-decoration: underline; color: blue;">
           <b>DreamZero Supervised Fine-Tuning</b>
         </a><br>
         Full-parameter and mixture SFT for DreamZero (WAN2.1 / WAN2.2 backbones)
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/release_0.2/qwen2_5_sft_vlm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_vlm.html" style="text-decoration: underline; color: blue;">
           <b>VLM Supervised Fine-Tuning</b>
         </a><br>
         Run full-parameter SFT and evaluation for VLM models such as Qwen
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   OpenPI <embodied/sft_openpi>
   OpenPI_mixed_precision <embodied/sft_openpi_pytorch>
   DreamZero <embodied/sft_dreamzero>
   VLM <embodied/sft_vlm>
