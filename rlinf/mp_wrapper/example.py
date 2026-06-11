
    # Apply MixedPrecisionWrapper AFTER FSDP wrapping so hooks fire on
    # FSDP-managed params (FlatParameter for FSDP1, DTensor for FSDP2).
    if use_wrapper:
        model = MixedPrecisionWrapper(
            model,
            compute_dtype=torch.bfloat16,
            keep_master_weights_on_cpu=master_cpu,
            stream_grads_to_cpu=stream_grads,
        )
        if is_main:
            logging.info(
                f"MixedPrecisionWrapper: master_cpu={master_cpu}, stream_grads={stream_grads}"
            )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    optim_kwargs = dict(
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    if isinstance(model, MixedPrecisionWrapper):
        wrapper_optim = os.environ.get("WRAPPER_OPTIM", "cpuadam")
        if wrapper_optim == "cpuadam":
            from hybrid_adam import CPUAdam
            optim = WrappedOptimizer(
                CPUAdam, model,
                **optim_kwargs,
                adamw_mode=True,
            )
        else:
            optim = WrappedOptimizer(
                torch.optim.AdamW, model,
                **optim_kwargs,
                fused=True,
            )
        if is_main:
            logging.info(f"Wrapper optimizer: {wrapper_optim}")
    else:
        optim = torch.optim.AdamW(
            model.parameters(),
            **optim_kwargs,
            fused=True,
        )
