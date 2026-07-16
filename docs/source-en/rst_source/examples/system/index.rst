System-level Optimizations
==============================

RLinf's overall design is simple and modular.
Workers abstract components for RL and agents, with a flexible and efficient communication library enabling inter-component interaction.
Thanks to this decoupled design, workers can be flexibly and dynamically scheduled to computing resources or assigned to the most suitable accelerators.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]Hot Scaling/Switching of Workers (Components)</b><br>
         Hot switching reduces training time by 50%+
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[Ongoing]Hybrid Training on Heterogeneous Accelerator</b><br>
         Flexible inter-operability between components on different accelerators to build training workflows
       </p>
     </div>
   </div>


.. toctree::
   :hidden:
   :maxdepth: 2

   fusco
