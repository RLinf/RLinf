系统级优化
============

RLinf的整体设计简洁且模块化，以Worker为抽象封装强化学习训练、智能体中的组件，提供灵活高效的通信库做组件间通信。基于这种解耦的设计，可以灵活调度Worker所使用的计算资源，也可以将Worker分配到更适配的加速器上。

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 20px; align-items: flex-start; flex-wrap: wrap;">
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]Worker(组件)间秒级热切换</b><br>
         秒级热切换提升训练速度50%+
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/waiting_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <b>[开发中]异构加速器混合训练</b><br>
         使用不同加速器运行的组件间灵活互通，构建训练工作流
       </p>
     </div>
   </div>