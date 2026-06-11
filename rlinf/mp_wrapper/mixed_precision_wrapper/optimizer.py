"""
WrappedOptimizer: wraps a torch optimizer to transparently handle
MixedPrecisionWrapper's grad management.

Usage:
    model = MixedPrecisionWrapper(raw_model, ...)
    optim = WrappedOptimizer(torch.optim.AdamW, model, lr=1e-3, ...)
    # optim.step() directly steps on fp32 master params.
    # optim.zero_grad() zeros master.grad and syncs master->compute.
"""

import typing

from rlinf.mp_wrapper.hybrid_adam.cpu_adam import CPUAdam
from rlinf.mp_wrapper.mixed_precision_wrapper.wrapper import MixedPrecisionWrapper

_T = typing.TypeVar("_T")
def WrappedOptimizer(optim_cls: type[_T], model, *args, **kwargs):
    """Create a wrapped optimizer for use with MixedPrecisionWrapper.

    Args:
        optim_cls: torch optimizer class (e.g. torch.optim.AdamW).
        model: MixedPrecisionWrapper instance. Its parameters() returns
               fp32 master params with grad already set by backward hooks.
        *args, **kwargs: passed to optim_cls.

    Returns:
        An optimizer instance whose zero_grad() triggers master->compute sync.
    """
    assert isinstance(model, MixedPrecisionWrapper)

    if issubclass(optim_cls, CPUAdam):
        class _Wrapper(optim_cls):
            def zero_grad(self, *a, **kw):
                model.on_optimizer_pre_zero_grad()
                result = super().zero_grad(*a, **kw)
                model.on_optimizer_post_zero_grad()
                return result

            def step(self, *args, **kwargs):
                model.on_optimzer_pre_step()
                super().step(*args, **kwargs)

            def _post_update(self, param, *state_keys: str) -> None:
                model.on_optimzer_post_step_update(param)

    else:
        class _Wrapper(optim_cls):
            def zero_grad(self, *a, **kw):
                model.on_optimizer_pre_zero_grad()
                result = super().zero_grad(*a, **kw)
                model.on_optimizer_post_zero_grad()
                return result

            def step(self, *args, **kwargs):
                model.on_optimzer_pre_step()
                super().step(*args, **kwargs)

    return typing.cast(_T, _Wrapper(model.parameters(), *args, **kwargs))
