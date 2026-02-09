Online RL for Code Completion Agent
=======================================================

Online Reinforcement Learning for Code Completion Agent is an important application scenario in the RLinf framework.
Through integration with code editors like Continue, we can collect user preference feedback on code completions, enabling near real-time code generation and feedback learning to quickly improve code completion quality and align with user preferences.
This example demonstrates how to use the RLinf framework to train a model capable of online code completion tasks.

Related reading: :doc:`A First Look at the "Last Mile" of Agent Deployment: Cursor Online Reinforcement Learning <../../blog/build_a_coding_online_rl_case>`.

Overview
--------

The online reinforcement learning for code completion agent system works through the following process:

1. **Real-time Interaction**: The system receives code completion requests from editors like Continue
2. **Model Inference**: Uses trained models to generate code completion suggestions
3. **User Feedback**: Collects user acceptance/rejection feedback on generated code
4. **Online Learning**: Updates model parameters in real-time based on user feedback

This real-time learning mechanism allows the model to quickly adapt to user programming habits and preferences.

Running the Script
------------------

**Environment Setup**

First, ensure you have installed the RLinf framework and its dependencies:

.. code-block:: bash

   # Install additional dependencies
   pip install httpx asyncio

If using the offline validation example, download the dataset:

.. code-block:: bash

   modelscope download --dataset "paxionfruit/code-fim-v2-python-filtered" --local_dir code-fim-v2-python-filtered

**Configure Continue Integration**

1. **Install Continue Extension**
   
   Since the current Continue does not support uploading user preference feedback on code completions, we have modified the Continue source code to support uploading user preference feedback on code completions.
   Users can get the compiled modified Continue plugin from `here <https://github.com/RLinf/continue/releases>`_ or build it themselves.

   After downloading the compiled Continue plugin, install it in VS Code.

   Method 1: code --install-extension /path/to/continue-1.3.9.vsix"

   Method 2: In VSCode, press Cmd+Shift+P, type 'Extensions: Install from VSIX', and select the above file

2. **Configure Continue Settings**

   The Continue configuration file path is:

   .. code-block:: bash

      ~/.continue/config.yaml

   Add the following settings to your Continue configuration file:

   .. code-block:: yaml

      # Please replace http://xxx:xx/ with the actual RLinf online code completion service address

      # Add a model for code completion
      models:
        - name: my-autocomplete
          provider: openai
          model: Qwen2.5-Coder-1.5B
          apiBase: http://xxx:8081/v1
          apiKey: xxx
          roles:
            - autocomplete

      # Add sending user feedback on whether to accept code completions
      tabAutocompleteOptions:
        enableCompletionTracking: true
        completionTrackingUrl: http://xxx:8082/api/training/submit
        completionTrackingHeaders:
          Authorization: Bearer test-token
          X-Project-ID: test-project
        maxPromptTokens: 1024
        debounceDelay: 350
        multilineCompletions: auto

   After modifying and saving, open the Continue extension from the left panel, click the "Settings" gear button in the top right corner, and ensure "Autocomplete Model" is set to my-autocomplete in the "Models" page.

**Start Training Service**

1. **Prepare Model and Configuration**

   - For online RL, edit and use `examples/coding_online_rl/config/qwen2.5-1.5b-ppo.yaml`:

     .. code-block:: yaml

        runner:
          output_dir: /path/to/your/logs

        rollout:
          model:
            model_path: /path/to/your/model

   - For offline validation, edit and use `examples/coding_online_rl/config/qwen2.5-1.5b-grpo-llm_judge.yaml`:

     .. code-block:: yaml

        runner:
          output_dir: /path/to/your/logs

        rollout:
          model:
            model_path: /path/to/your/model

        data:
          train_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_train_3k.jsonl"]
          val_data_paths: ["/path/to/your/dataset/code-fim-v2-python-filtered_formatted_test_1k.jsonl"]

     Also set the API endpoint and key for the LLM-as-judge used to simulate feedback:

     .. code-block:: bash

        export LLMASJUDGE_API_URL=your_api_url
        export LLMASJUDGE_API_KEY=your_api_key
        export LLMASJUDGE_MODEL=your_model  # not recommended; the prompt should fit your model.

2. **Start RLinf Training Service**
   
   - For online RL:

     .. code-block:: bash

        # Navigate to project directory
        cd /path/to/rlinf_online_rl

        # Start training service
        bash examples/coding_online_rl/run_main_coding_online_rl.sh

     This will start the following services:
     - **Inference Service**: Provides code completion API on port 8081
     - **Training Service**: Receives user feedback data on port 8082

   - For offline validation:

     .. code-block:: bash

        # Navigate to project directory
        cd /path/to/rlinf_online_rl

        # Start training service
        bash examples/coding_online_rl/run_main_coding_rl_llm_judge.sh

**Integration with Continue**

1. **Start Continue**
   
   Launch the Continue extension in VS Code, ensuring it connects to the correct API endpoints.

2. **Begin Programming**
   
   Start writing code in Continue. The system will:
   - Automatically send code completion requests to the inference service
   - Receive model-generated code suggestions
   - Collect your acceptance/rejection feedback on suggestions

3. **Real-time Learning**
   
   The system processes your feedback in real-time:
   - Accepted suggestions are marked as positive feedback
   - Rejected suggestions are marked as negative feedback
   - Model parameters are updated online based on feedback

**Monitor Training Process**

You can monitor the training process through the following methods:

1. **View Log Output**
   
   .. code-block:: bash

      # View training logs
      tail -f results/ppo-1.5b/train.log

2. **Use TensorBoard**
   
   .. code-block:: bash

      # Start TensorBoard
      tensorboard --logdir results/grpo-1.5b

3. **Check Model Checkpoints**
   
   Model checkpoints are periodically saved to the `results/grpo-1.5b/checkpoints/` directory during training.

**Test Client**

You can use the provided test client to verify system functionality:

.. code-block:: bash

   # Run test client
   python examples/coding_online_rl/simple_test_client.py

The test client simulates Continue behavior by sending code completion requests and submitting feedback data.

**Troubleshooting**

Common issues and solutions:

1. **Port Conflicts**
   
   If ports 8081 or 8082 are occupied, modify the port settings in the configuration file.

2. **Model Loading Failure**
   
   Check that the model path is correct and ensure model files exist and are accessible.

3. **Continue Connection Failure**
   
   Ensure the API endpoint addresses in Continue configuration are correct and check network connectivity. You can also use simple_test_client to test if feedback data can be received normally.

Through these steps, you can successfully run the online reinforcement learning for code completion agent system and achieve seamless integration with the Continue editor.
