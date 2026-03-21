# **Explainability**

|Field:|Response:|  
|:---:|:---:|  
|Intended Domain:| Open foundation model for generalized humanoid robot reasoning and skills.|  
|Model Type: |Robot VLA model|  
|Intended Users:|This model is intended for developers and community that build and finetune robot foundation models.|  
|Output:|The model outputs are actions, and the units are floating points. This is referred to as "robot action policy." Actions consist of continuous-value vectors that correspond to different motor controls on a robot.|  
|Describe how the model works:|Accepts vision, language and proprioception, outputs robot action policy.|  
|Technical Limitations & Mitigation:| This model is not tested or intended for use in mission critical applications that require functional safety. The use of the model in those applications is at the user's own risk and sole responsibility, including taking the necessary steps to add needed guardrails or safety mechanisms prior to deployment.<br><br>Risk: Model underperformance in highly dynamic environments with varying robot surroundings (e.g. furniture, objects, etc) and lighting conditions.<br>Mitigation: Enhance dataset with dynamic obstacle scenarios and fine-tune models accordingly.<br><br>Risk: Integration challenges in specific customer environments with varying robot surroundings (e.g. furniture, objects, etc) and lighting conditions.<br>Mitigation: Provide detailed integration guides and support, leveraging NVIDIA's ecosystem.<br><br>Risk: Limited initial support for certain robot embodiments.<br>Mitigation: Expand testing and validation across a wider range of robot platforms.|  
|Verified to have met prescribed quality standards?|Yes|  
|Performance Metrics:|Success rate, as well as the following:<br>1) if the trajectory is smooth and does not jitter<br>2) if the robot does not hit any other objects<br>3) if the trajectory is natural|  
|Potential Known Risks:|This model is not tested or intended for use in mission critical applications that require functional safety. The use of the model in those applications is at the user's own risk and sole responsibility, including taking the necessary steps to add needed guardrails or safety mechanisms prior to deployment.|  
|End User License Agreement:| Your use of this model is governed by the [NSCL V1 License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyIsIm5jaWQiOiJzby15b3V0LTg3MTcwMS12dDQ4In0=).|