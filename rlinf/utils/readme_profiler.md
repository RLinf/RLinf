# Use PyTorch Profiler in Megatron when inference and training

`rlinf/utils/profiler.py` defines utils to profile megatron when inference and training. 

This document provides an overview of the two main classes in this file: `PyTorchProfiler` and `PyTorchProfilerFunc`.

### `PyTorchProfilerFunc`

The `PyTorchProfilerFunc` class is a lightweight wrapper around `torch.profiler.record_function`. It primarily implements `start()` and `stop()` methods to conditionally activate a `record_function` context.

This design improves code clarity and cleanliness. By using this wrapper, you can embed profiling markers in your code without needing to add `if profiler_enabled:` checks everywhere, as the conditional logic is handled internally by the class.

### `PyTorchProfiler`

The `PyTorchProfiler` class is a comprehensive wrapper for the main `torch.profiler.profile` functionality. It provides `start()` and `stop()` methods designed to profile specific, critical code blocks, such as the `run_forward_backward` method within the `MegatronActor` class. This is particularly useful as this method is invoked during both training and inference phases.

The profiler's behavior is configured through a dedicated section in the YAML configuration file. To enable the profiler, the `use_profiler` flag must be set to `True`. Note that enabling the profiler will have a performance impact.

#### Configuration

The profiler is configured under the `megatron.profiler` key in your YAML file:

```yaml
megatron:
  # ...
  use_profiler: False # If true, enables the torch profiler. Be aware of the performance overhead.
  # ...
      
  profiler: # Configuration for Megatron profiling during inference and training
    output_dir: ${runner.output_dir}/${runner.experiment_name}/profiler
    activities: ["cpu", "cuda"]
    record_shapes: False
    profile_memory: False
    with_stack: False
    with_flops: False
    with_modules: True
    export_tensorboard: True
    export_chrome_trace: False
    chrome_filename_prefix: "chrome_trace"
    schedule_warmup: 2
    schedule_active: 1
    schedule_repeat: 1 # The profiling cycle will repeat this many times for both training and inference
    # schedule_wait: This value is set dynamically at runtime.
```

#### Parameter Descriptions

Here is a detailed explanation of each configuration parameter:

*   **`output_dir`**: The directory where the profiler output files (TensorBoard traces, Chrome traces) will be saved.
*   **`activities`**: A list specifying the hardware activities to profile.
    *   `"cpu"`: Enables profiling of CPU operations.
    *   `"cuda"`: Enables profiling of CUDA kernel executions.
*   **`record_shapes`**: If `True`, records the shapes of operator inputs. This is useful for debugging but adds overhead.
*   **`profile_memory`**: If `True`, enables memory profiling, tracking memory allocations and releases on both CPU and CUDA devices. This adds significant overhead.
*   **`with_stack`**: If `True`, records the Python source code call stack for each operation. This is extremely useful for identifying the origin of an operation in your code but comes with a high performance cost.
*   **`with_flops`**: If `True`, attempts to estimate the FLOPs for each operation, helping to analyze model compute characteristics.
*   **`with_modules`**: If `True`, associates profiled operations with their corresponding `torch.nn.Module` hierarchy, making it easier to identify which part of your model is responsible for a given operation.
*   **`export_tensorboard`**: If `True`, generates a trace file that can be loaded and visualized in TensorBoard.
*   **`export_chrome_trace`**: If `True`, generates a JSON file that can be viewed in Chrome's tracing tool (`chrome://tracing`), providing a detailed timeline of events.
*   **`chrome_filename_prefix`**: The prefix for the generated Chrome trace JSON file.
*   **`schedule_warmup`**: The number of steps to run for warmup. Operations in these steps are profiled but not recorded in the final trace. This helps to mitigate one-off costs like CUDA context creation and ensures the profiled workload is in a steady state.
*   **`schedule_active`**: The number of steps to actively record. The traces from these steps are saved to the specified `output_dir`.
*   **`schedule_repeat`**: The number of times the complete `wait -> warmup -> active` cycle should be repeated.


### Function Details

There are three exposed functions in `PyTorchProfiler`:

- `init_fwd_bwd_schedule(self, num_minibatches)`: it will init training's schedule info using parameter `num_minibatches`，
    setting `schedule_wait` by `num_minibatches` -  `self.schedule_warmup` - `self.schedule_active` and check whether it's
    a legal value.

- `start(self, forward_only: bool = False)`: it will start according profiler by parameter `forward_only`, if True will profiler
    inference, otherwise will profile training.

- `stop(self, forward_only: bool = False)`: it will stop former used profiler created by `start` function, and advance according step by parameter `forward_only`.

#### Important Considerations

*   The `schedule_wait` parameter, which defines the number of initial steps to skip before the warmup phase begins, is **calculated dynamically at runtime** and is not set in the YAML file.
*   The profiling cycle will be executed `schedule_repeat` times for **both training and inference**.
*   The profiling schedule behaves differently for inference. During inference, each profiled call is treated as an independent cycle where **`wait` and `warmup` are 0, and `active` is 1**. This ensures that individual inference requests are captured immediately without any warmup or delay.


### Results Usage

`Parameter Descriptions` section tells profile results will be put into selected `output_dir`, more detaily, it will generate `fwd` and `fwd_bwd` folder in `output_dir`, the former is for inference,  the latter is for training. If you set `export_tensorboard` True, plase see 
[Pytorch Profiler Tutorial](https://docs.pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) for detailed usage.

