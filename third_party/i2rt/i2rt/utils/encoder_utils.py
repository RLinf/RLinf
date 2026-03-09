from typing import Any, Callable

from i2rt.motor_drivers.dm_driver import CanInterface, EncoderChain, PassiveEncoderReader


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    """Return an EncoderChain that shares the same CAN bus with DM motors.

    This helper is used by DMChainCanInterface via the ``get_same_bus_device_driver``
    argument so that the passive encoder (ID 0x50E) can be polled in the same
    control loop without opening another CAN channel.

    Parameters
    ----------
    can_interface : CanInterface
        The CAN interface instance internally created by ``DMChainCanInterface``.

    Returns
    -------
    EncoderChain
        A chain that contains the single passive encoder on ID ``0x50E``.
    """
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    # The passive encoder report frame is sent with ID 0x50E+receive_mode offset.
    # We only have one encoder in this setup, hence the singleton list.
    return EncoderChain([0x50E], passive_encoder_reader)


def make_get_encoder_chain() -> Callable[[Any], EncoderChain]:
    """Factory that returns the *callable* ``get_encoder_chain`` itself.

    Used in YAML so that the config instantiate step invokes this factory with
    no arguments, giving back the callable which will later receive the
    ``can_interface`` from ``DMChainCanInterface`` at runtime.
    """
    return get_encoder_chain
