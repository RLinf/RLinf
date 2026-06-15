VLA / WAM 模型监督微调
======================

监督微调（SFT）是具身强化学习的标准冷启动步骤：一个良好的 SFT 检查点能显著缩短 RL 探索时间并提升最终策略效果。本类示例汇总了 RLinf 在 VLA / WAM 模型上的全量与 LoRA SFT 配方，以及面向多模态后训练的 VLM SFT。

完成本节的 SFT 后，可继续阅读 :doc:`vla_wam_index`\ （以模型为主线的 RL 微调）或 :doc:`simulators_index`\ （以基准为主线的 RL 微调）以进一步用 RL 提升所得检查点。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <!-- TODO(thumbnail): replace placeholder cover image URL for sft_openpi -->
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_openpi.html" style="text-decoration: underline; color: blue;">
           <b>OpenPI 监督微调</b>
         </a><br>
         支持 OpenPI 全量 SFT 与 LoRA 微调，作为强化学习前置阶段
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://dreamzero0.github.io/images/project_overview.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_dreamzero.html" style="text-decoration: underline; color: blue;">
           <b>DreamZero 监督微调</b>
         </a><br>
         面向 DreamZero 的全量与 mixture SFT（WAN2.1 / WAN2.2 主干）
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/release_0.2/qwen2_5_sft_vlm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/sft_vlm.html" style="text-decoration: underline; color: blue;">
           <b>VLM模型监督微调训练</b>
         </a><br>
         支持 Qwen 系列等 VLM 的全量监督微调与结果评估
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   embodied/sft_openpi
   embodied/sft_dreamzero
   embodied/sft_vlm
