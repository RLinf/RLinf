On Ascend 950 NPUs, add ``--torch 2.11.0`` before ``embodied`` in the native
installation command. The Ascend installer uses PyTorch and ``torch-npu`` 2.6.0
by default, and ``torch-npu`` 2.6.0 fails to initialize a 950 NPU with
``Unsupported soc version``. PyTorch and ``torch-npu`` 2.11.0 have been verified
on Ascend 950 with NPU driver 25.7.rc1 and CANN 9.1.1.
