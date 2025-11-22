# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

from ..hardware import AcceleratorType, AcceleratorUtil
from .async_work import AsyncCollWork, AsyncWork
from .collective_group import CollectiveGroup, CollectiveGroupInfo


class MultiChannelProcessGroup:
    """A wrapper class for multiple dist.ProcessGroup that supports multi-channel communication.

    This class offers send/recv APIs that accepts channel_id to specify which channel to use for communication.

    Args:
        cur_rank (int): The current rank in the group.
        num_channels (int): The number of channels to use for communication.
        logger (Optional[logging.Logger]): Optional logger for debugging.

    """

    def __init__(
        self,
        cur_rank: int,
        num_channels: int,
        group_info: CollectiveGroupInfo,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the MultiChannelProcessGroup.

        Args:
            cur_rank (int): The current rank in the group.
            num_channels (int): The number of channels to use for communication.
            group_info (CollectiveGroupInfo): The collective group information.
            logger (Optional[logging.Logger]): Optional logger for debugging.

        """
        self._cur_rank = cur_rank
        self._peer_rank = 1 if cur_rank == 0 else 0
        self._num_channels = num_channels
        self._logger = logger
        self._is_initialized = False
        self._no_accel_ccl = False
        self._group_name = None
        self._group_info = group_info

        # Check if all workers have the same accelerator type
        accel_type = group_info.workers[0].accelerator_type
        self._no_accel_ccl = (
            # Hetero workers in the same group, disable CCL
            any(worker.accelerator_type != accel_type for worker in group_info.workers)
            # CPU only, disable CCL
            or accel_type == AcceleratorType.NO_ACCEL
            # Unsupported accelerator CCL type, disable CCL
            or accel_type not in AcceleratorUtil.CCL_SUPPORT_LIST
        )
        self._accel_ccl_backend = (
            AcceleratorUtil.get_ccl_backend(accel_type)
            if not self._no_accel_ccl
            else None
        )

        self._send_accel_ccl_process_groups: list[dist.ProcessGroup] = [
            None for _ in range(num_channels)
        ]
        self._recv_accel_ccl_process_groups: list[dist.ProcessGroup] = [
            None for _ in range(num_channels)
        ]
        self._send_gloo_process_groups: list[dist.ProcessGroup] = [
            None for _ in range(num_channels)
        ]
        self._recv_gloo_process_groups: list[dist.ProcessGroup] = [
            None for _ in range(num_channels)
        ]

    @property
    def is_initialized(self) -> bool:
        """Check if the MultiChannelProcessGroup is initialized."""
        return self._is_initialized

    def init(
        self,
        init_method: str,
        world_size: int,
        rank: int,
        group_name: str,
    ):
        """Initialize a MultiChannelProcessGroup. The parameters are of the same meaning as the torch.distributed.init_process_group function.

        Args:
            init_method (str): The initialization method for the process group.
            world_size (int): The total number of processes in the group.
            rank (int): The rank of the current process in the group.
            group_name (str): The name of the group.

        """
        from ..cluster import Cluster, ClusterEnvVar

        self._group_name = group_name
        try:
            # Set default timeout to 180 minutes
            timeout = int(Cluster.get_sys_env_var(ClusterEnvVar.TIMEOUT, "180"))
            self._logger.debug(
                f"Setting timeout to {timeout} minutes for group {group_name}"
            )
            timeout = timedelta(minutes=timeout)
        except ValueError:
            raise ValueError(
                "Invalid TIMEOUT value. It should be an integer representing minutes."
            )

        if not self._no_accel_ccl:
            # Create accelerator CCL groups and split GLOO groups from them
            base_group = MultiChannelProcessGroup._create_process_group(
                backend=self._accel_ccl_backend,  # Only NCCL group supports splitting
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                group_name=group_name + f"{self._accel_ccl_backend}_send_0",
                timeout=timeout,
                # device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
                # Setting device_id is crucial triggers eager creation of NCCL communicators
                # https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
                # If not, communicators will only be created upon the first collective operation
                # If the first pair of communications are from different process groups (e.g., two async recvs from a group), the NCCL group creation will hang by then
                # However, eager creation of NCCL communicators leads to severe GPU memory consumption. So we disable it by default.
            )

            for i in range(self._num_channels):
                self._send_accel_ccl_process_groups[i] = (
                    MultiChannelProcessGroup._split_process_group(
                        base_group=base_group,
                        backend=self._accel_ccl_backend,
                        group_name=group_name + f"{self._accel_ccl_backend}_send_{i}",
                        timeout=timeout,
                    )
                    if i > 0
                    else base_group
                )

                self._recv_accel_ccl_process_groups[i] = (
                    MultiChannelProcessGroup._split_process_group(
                        base_group=base_group,
                        backend=self._accel_ccl_backend,
                        group_name=group_name + f"{self._accel_ccl_backend}_recv_{i}",
                        timeout=timeout,
                    )
                )

                self._send_gloo_process_groups[i] = (
                    MultiChannelProcessGroup._split_process_group(
                        base_group=base_group,
                        backend="gloo",
                        group_name=group_name + f"gloo_send_{i}",
                        timeout=timeout,
                    )
                )

                self._recv_gloo_process_groups[i] = (
                    MultiChannelProcessGroup._split_process_group(
                        base_group=base_group,
                        backend="gloo",
                        group_name=group_name + f"gloo_recv_{i}",
                        timeout=timeout,
                    )
                )
        else:
            # Create only GLOO groups when accelerator CCL is not available
            # GLOO does not support splitting, only reuse its store
            base_group = MultiChannelProcessGroup._create_process_group(
                backend="gloo",
                init_method=init_method,
                world_size=world_size,
                rank=rank,
                group_name=group_name + "gloo_send_0",
                timeout=timeout,
            )
            base_store = torch.distributed.distributed_c10d._get_process_group_store(
                base_group
            )

            for i in range(self._num_channels):
                self._send_gloo_process_groups[i] = (
                    MultiChannelProcessGroup._create_process_group(
                        backend="gloo",
                        world_size=world_size,
                        rank=rank,
                        store=base_store,
                        group_name=group_name + f"gloo_send_{i}",
                        timeout=timeout,
                    )
                    if i > 0
                    else base_group
                )

                self._recv_gloo_process_groups[i] = (
                    MultiChannelProcessGroup._create_process_group(
                        backend="gloo",
                        world_size=world_size,
                        rank=rank,
                        store=base_store,
                        group_name=group_name + f"gloo_recv_{i}",
                        timeout=timeout,
                    )
                )

        if self._cur_rank == 1:
            # Swap send and recv process groups if the current rank is the last rank
            self._send_accel_ccl_process_groups, self._recv_accel_ccl_process_groups = (
                self._recv_accel_ccl_process_groups,
                self._send_accel_ccl_process_groups,
            )
            self._send_gloo_process_groups, self._recv_gloo_process_groups = (
                self._recv_gloo_process_groups,
                self._send_gloo_process_groups,
            )

        self._is_initialized = True

    def send(
        self, tensor: torch.Tensor, device: str, channel_id: int, async_op: bool = False
    ) -> Optional[AsyncWork]:
        """Send a tensor via a channel.

        Args:
            tensor (torch.Tensor): The tensor to send.
            device (str): The device type, either CollectiveGroup.CUDA or CollectiveGroup.GLOO.
            channel_id (int): The channel ID to use for sending the tensor.
            async_op (bool): Whether to perform the operation asynchronously.

        """
        if not self._is_initialized:
            raise RuntimeError("MultiChannelProcessGroup is not initialized")
        if channel_id < 0 or channel_id >= self._num_channels:
            raise ValueError(
                f"Invalid channel_id: {channel_id}. Must be in range [0, {self._num_channels - 1}]"
            )

        # NOTE: GLOO backend doesn't support dist.Work.get_future, use broadcast to simulate send/recv instead
        if self._no_accel_ccl and device == CollectiveGroup.ACCEL:
            raise RuntimeError(
                f"Collective group {self._group_name} does not support accelerator CCL backend, possibly because (1) the workers in the group have different accelerator types:  {[worker.accelerator_type for worker in self._group_info.workers]}, (2) the workers are CPU-only, or (3) the accelerator CCL is not among the supported CCL: {AcceleratorUtil.CCL_SUPPORT_LIST}."
            )
        group = (
            self._send_accel_ccl_process_groups[channel_id]
            if device == CollectiveGroup.ACCEL
            else self._send_gloo_process_groups[channel_id]
        )
        work = self._broadcast(
            tensor,
            src=self._cur_rank,
            group=group,
            async_op=async_op,
        )
        if work:
            return AsyncCollWork(work)

    def recv(
        self, tensor: torch.Tensor, device: str, channel_id: int, async_op: bool = False
    ) -> Optional[AsyncWork]:
        """Receive a tensor from a peer rank.

        Args:
            tensor (torch.Tensor): The tensor to receive.
            device (str): The device type, either CollectiveGroup.CUDA or CollectiveGroup.GLOO.
            channel_id (int): The channel ID to use for receiving the tensor.
            async_op (bool): Whether to perform the operation asynchronously.

        """
        if not self._is_initialized:
            raise RuntimeError("MultiChannelProcessGroup is not initialized")
        if channel_id < 0 or channel_id >= self._num_channels:
            raise ValueError(
                f"Invalid channel_id: {channel_id}. Must be in range [0, {self._num_channels - 1}]"
            )

        # NOTE: GLOO backend doesn't support dist.Work.get_future, use broadcast to simulate send/recv instead
        group = (
            self._recv_accel_ccl_process_groups[channel_id]
            if device == CollectiveGroup.ACCEL
            else self._recv_gloo_process_groups[channel_id]
        )
        work = self._broadcast(
            tensor,
            src=self._peer_rank,
            group=group,
            async_op=async_op,
        )

        if async_op:
            return AsyncCollWork(work)

    def _broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: dist.ProcessGroup = None,
        async_op: bool = False,
    ):
        """Broadcast a tensor in the given process group.

        This is modified version of dist.broadcast to avoid checking default group both in the broadcast and in the _exception_logger annotator's get_msg_dict function.
        """
        try:
            from torch.distributed.distributed_c10d import (
                BroadcastOptions,
                _check_single_tensor,
                _rank_not_in_group,
                _warn_not_in_group,
                get_group_rank,
            )

            _check_single_tensor(tensor, "tensor")
            if _rank_not_in_group(group):
                _warn_not_in_group("broadcast")
                return

            opts = BroadcastOptions()
            opts.rootRank = src
            opts.rootTensor = 0
            opts.asyncOp = async_op

            if group is None:
                raise ValueError("Group must be specified for broadcast operation")
            group_src_rank = get_group_rank(group, src)
            opts.rootRank = group_src_rank
            work = group.broadcast([tensor], opts)
            if async_op:
                return work
            elif work is not None:
                work.wait()
        except Exception as error:
            pg_name = dist._get_process_group_name(group)
            msg = f"Broadcast failed on ProcessGroup {pg_name} rank {self._cur_rank} with error: {error}. Args - tensor: {tensor}, src: {src}, group: {group}, async_op: {async_op}."
            self._logger.error(msg)

    @staticmethod
    def _create_process_group(
        backend=None,
        init_method=None,
        timeout=None,
        world_size=-1,
        rank=-1,
        store=None,
        group_name=None,
        pg_options=None,
        device_id=None,
    ) -> dist.ProcessGroup:
        """Create a new process group.

        This function is modified version of dist.init_process_group to allow creating multiple process groups without the default process group and new processes unknown to existing process groups (therefore new_group cannot be used).
        """
        from torch.distributed.distributed_c10d import (
            Backend,
            PrefixStore,
            _world,
            default_pg_timeout,
            rendezvous,
        )

        assert (store is None) or (init_method is None), (
            "Cannot specify both init_method and store."
        )

        if store is not None:
            assert world_size > 0, "world_size must be positive if using store"
            assert rank >= 0, "rank must be non-negative if using store"
        elif init_method is None:
            init_method = "env://"

        if backend:
            backend = Backend(backend)
        else:
            backend = Backend("undefined")

        if timeout is None:
            timeout = default_pg_timeout

        # backward compatible API
        if store is None:
            rendezvous_iterator = rendezvous(
                init_method, rank, world_size, timeout=timeout
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timeout)

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore(group_name, store)

        pg, _ = MultiChannelProcessGroup._new_process_group_helper(
            None,
            world_size,
            rank,
            [],
            backend,
            store,
            group_name=group_name,
            pg_options=pg_options,
            device_id=device_id,
            timeout=timeout,
        )

        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

        return pg

    @staticmethod
    def _split_process_group(
        base_group: dist.ProcessGroup = None,
        timeout=None,
        backend=None,
        pg_options=None,
        group_name=None,
    ) -> dist.ProcessGroup:
        """Split an existing process group.

        This is a modified version of dist.new_group to allow splitting any existing process groups (not only the default group like dist.new_group) into a new one.
        """
        dist.new_group
        from torch.distributed.distributed_c10d import (
            Backend,
            _check_valid_timeout,
            _get_default_timeout,
            _process_group_name,
            _world,
        )

        default_pg = base_group
        device_id = default_pg.bound_device_id
        default_backend, default_store = _world.pg_map[default_pg]
        global_rank = default_pg.rank()
        global_world_size = default_pg.size()

        # Default to the same backend as the global process group
        # if the backend is not specified.
        if not backend:
            backend = default_backend
        backend = Backend(backend)

        # this timeout defaulting/validation is used for all the new_groups/new_subgroups variants,
        # which may just pass their timeout value (or None)
        if timeout is None:
            timeout = _get_default_timeout(backend)
        _check_valid_timeout(timeout)

        # checks the input ranks
        ranks = list(range(global_world_size))
        group_world_size = global_world_size
        group_rank = global_rank

        if group_name is None:
            group_name = _process_group_name(ranks, use_hashed_name=False)

        pg, _ = MultiChannelProcessGroup._new_process_group_helper(
            base_group,
            group_world_size,
            group_rank,
            ranks,
            backend,
            default_store,
            group_name=group_name,
            pg_options=pg_options,
            timeout=timeout,
            device_id=device_id,
        )

        # Create the global rank to group rank mapping
        _world.pg_group_ranks[pg] = {
            global_rank: group_rank for group_rank, global_rank in enumerate(ranks)
        }

        return pg

    @staticmethod
    def _new_process_group_helper(
        base_group,
        group_size,
        group_rank,
        global_ranks_in_group,
        backend,
        store,
        group_name,
        pg_options=None,
        timeout=None,
        pg_tag=None,
        device_id=None,
        group_desc=None,
    ):
        """Help create the process group.

        This is modified version of dist._new_process_group_helper that allows splitting any existing process groups (not only the default group like dist._new_process_group_helper) into a new one. Check MODIFICATION NOTE comment in the function.
        """
        from torch.distributed.distributed_c10d import (
            _GLOO_AVAILABLE,
            _MPI_AVAILABLE,
            _NCCL_AVAILABLE,
            _UCC_AVAILABLE,
            Backend,
            BackendConfig,
            DebugLevel,
            GroupMember,
            PrefixStore,
            ProcessGroup,
            _check_valid_timeout,
            _create_process_group_wrapper,
            _DistributedBackendOptions,
            _find_pg_by_ranks_and_tag,
            _process_group_color,
            _register_process_group,
            _world,
            get_debug_level,
            is_mpi_available,
            is_nccl_available,
            is_ucc_available,
            logger,
        )

        if _GLOO_AVAILABLE:
            from torch.distributed.distributed_c10d import (
                ProcessGroupGloo,
                _ProcessGroupWrapper,
            )
        if _NCCL_AVAILABLE:
            from torch.distributed.distributed_c10d import ProcessGroupNCCL
        if _UCC_AVAILABLE:
            from torch.distributed.distributed_c10d import ProcessGroupUCC
        if _MPI_AVAILABLE:
            from torch.distributed.distributed_c10d import ProcessGroupMPI
        import warnings

        if group_name in _world.pg_names.values():
            raise ValueError(
                "The specified group name has already been "
                "created, please use a different group name"
            )

        if device_id is not None and (
            device_id.index is None or device_id.type != "cuda"
        ):
            raise ValueError(
                "init_process_group device_id parameter must be a cuda device with an "
                "id, e.g. cuda:0, not just cuda or cpu"
            )

        # Note: _new_process_group_helper is only called from init_process_group, which always provides a timeout value
        _check_valid_timeout(timeout)

        if pg_tag not in [None, ""]:
            # creating with the same tag and rank set results in the same underlying PG
            existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
            if existing_group:
                _, prefix_store = _world.pg_map[existing_group]
                return existing_group, prefix_store

        group_desc = "undefined" if group_desc is None else group_desc

        # The list of group ranks is empty if we're creating the default group.
        is_default_group = len(global_ranks_in_group) == 0

        # nccl and potentially other backends allow creation of
        # communicators based on pre-existing ones, which can save
        # initialization time.  Due to lazy initialization of
        # communicators in some backends, we have to be careful and only
        # split when we *know* the backends already are connected _on all
        # ranks_.  We can only know this if the group we are making is the
        # entire world or if we have bound a device id to the world (which
        # causes early connection initialization).
        # MODIFICATION NOTE: check default group -> check base group
        if base_group is not None and (
            len(global_ranks_in_group) == base_group.size()
            or base_group.bound_device_id
        ):
            split_from = None
            if base_group.bound_device_id:
                split_from = base_group._get_backend(base_group.bound_device_id)

            try:
                split_from = base_group._get_backend(torch.device("cuda"))
            except RuntimeError:
                # no cuda device associated with this backend
                pass

            if not split_from or not split_from.supports_splitting:
                return None

            # If necessary, find a backend to split from by peeling process
            # group wrappers from our potentially wrapped process group.
            while _GLOO_AVAILABLE and isinstance(split_from, _ProcessGroupWrapper):
                split_from = split_from.wrapped_pg
        else:
            split_from = None

        # If this is a subgroup (which means group_ranks is specified),
        # we check if the current process is a member of the new group.
        if not is_default_group:
            global_rank = base_group.rank()
            if global_rank not in global_ranks_in_group:
                # If we are using `ncclCommSplit` (or similar split from
                # other APIs) to create the communicator, we will need to
                # call `ncclCommSplit` on *all* ranks in this new group's
                # parent group, even those not in the new group.  This is
                # a requirement of the NCCL API as otherwise we would get
                # out of sync.
                if split_from:
                    split_from.perform_nocolor_split(base_group.bound_device_id)
                return GroupMember.NON_GROUP_MEMBER, None

        prefix_store = PrefixStore(f"{group_name}/", store)
        if hasattr(ProcessGroup, "Options"):
            # Torch 2.7 removed Options
            base_pg_options = ProcessGroup.Options(backend=str(backend))
            base_pg_options._timeout = timeout
            pg: ProcessGroup = ProcessGroup(
                prefix_store, group_rank, group_size, base_pg_options
            )
        else:
            pg: ProcessGroup = ProcessGroup(prefix_store, group_rank, group_size)
        if device_id:
            pg.bound_device_id = device_id
        backend_config = BackendConfig(backend)
        backend_class: torch._C._distributed_c10d.Backend
        for device, backend_str in backend_config.get_device_backend_map().items():
            # Use the group name as prefix in the default store, such that
            # a single store can be reused by multiple groups.
            backend_prefix_store = PrefixStore(f"{device}/", prefix_store)

            if backend_str == Backend.MPI:
                if not is_mpi_available():
                    raise RuntimeError(
                        "Distributed package doesn't have MPI built in."
                        " MPI is only included if you build PyTorch from"
                        " source on a host that has MPI installed."
                    )
                backend_class = ProcessGroupMPI.create(global_ranks_in_group)
                backend_type = ProcessGroup.BackendType.MPI
                if not backend_class:
                    return GroupMember.NON_GROUP_MEMBER, None
                # create new process group with accurate rank and size
                if pg.rank() == -1 and pg.size() == -1:
                    if hasattr(ProcessGroup, "Options"):
                        pg = ProcessGroup(
                            backend_prefix_store,
                            backend_class.rank(),
                            backend_class.size(),
                            base_pg_options,
                        )
                    else:
                        pg = ProcessGroup(
                            backend_prefix_store,
                            backend_class.rank(),
                            backend_class.size(),
                        )
            elif backend_str == Backend.GLOO:
                # TODO: remove this check after lazy initialization is supported
                # if pg_options is not None:
                #     raise RuntimeError("GLOO options not supported")
                backend_class = ProcessGroupGloo(
                    backend_prefix_store, group_rank, group_size, timeout=timeout
                )
                backend_type = ProcessGroup.BackendType.GLOO
            elif backend_str == Backend.NCCL:
                if not is_nccl_available():
                    raise RuntimeError("Distributed package doesn't have NCCL built in")
                if pg_options is not None:
                    assert isinstance(pg_options, ProcessGroupNCCL.Options), (
                        "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
                    )
                    if pg_options._timeout != timeout:
                        warnings.warn(
                            "pg_options._timeout was specified, "
                            "but timeout kwarg has a default value that will always override it. "
                        )
                else:
                    # default pg_options for NCCL
                    pg_options = ProcessGroupNCCL.Options()
                    pg_options.is_high_priority_stream = False
                pg_options._timeout = timeout

                if split_from:
                    pg_options.split_from = split_from
                    pg_options.split_color = _process_group_color(global_ranks_in_group)
                pg_options.global_ranks_in_group = global_ranks_in_group
                pg_options.group_name = group_name
                backend_class = ProcessGroupNCCL(
                    backend_prefix_store, group_rank, group_size, pg_options
                )
                backend_type = ProcessGroup.BackendType.NCCL
            elif backend_str == Backend.UCC and is_ucc_available():
                # TODO: once UCC plugin is fully deprecated, remove
                # is_ucc_available() from above elif-condition and raise
                # RuntimeError if is_ucc_available() returns false.

                backend_class = ProcessGroupUCC(
                    backend_prefix_store, group_rank, group_size, timeout=timeout
                )
                backend_type = ProcessGroup.BackendType.UCC
            else:
                assert backend_str.upper() in Backend._plugins, (
                    f"Unknown c10d backend type {backend_str.upper()}"
                )

                backend_plugin = Backend._plugins[backend_str.upper()]
                creator_fn = backend_plugin.creator_fn
                extended_api = backend_plugin.extended_api
                backend_type = ProcessGroup.BackendType.CUSTOM

                if not extended_api:
                    backend_class = creator_fn(
                        backend_prefix_store, group_rank, group_size, timeout
                    )
                else:
                    dist_backend_opts = _DistributedBackendOptions()
                    dist_backend_opts.store = backend_prefix_store
                    dist_backend_opts.group_rank = group_rank
                    dist_backend_opts.group_size = group_size
                    dist_backend_opts.timeout = timeout
                    dist_backend_opts.group_id = group_name
                    dist_backend_opts.global_ranks_in_group = global_ranks_in_group

                    backend_class = creator_fn(dist_backend_opts, pg_options)

            # Set sequence numbers for gloo and nccl backends.
            if backend_str == Backend.GLOO:
                assert isinstance(backend_class, ProcessGroupGloo)
                backend_class._set_sequence_number_for_group()
            elif backend_str == Backend.NCCL:
                assert isinstance(backend_class, ProcessGroupNCCL)
                backend_class._set_sequence_number_for_group()

            # If the type is a subclass of ProcessGroup then return this process group immediately
            # TODO: This defaults to the old behavior for PythonProcessGroups which overwrites the
            # ProcessGroup instance
            if issubclass(type(backend_class), ProcessGroup):
                pg = backend_class  # type: ignore[assignment]
                break

            # Process group wrapper initialization for supported PGs when TORCH_DISTRIBUTED_DEBUG is set
            if (
                backend_str in [Backend.GLOO, Backend.NCCL, Backend.UCC]
                or backend_str.upper() in Backend._plugins
            ):
                # In debug mode and if GLOO is available, wrap in a wrapper PG that
                # enables enhanced collective checking for debuggability.
                if get_debug_level() == DebugLevel.DETAIL:
                    if not _GLOO_AVAILABLE:
                        logger.info(
                            """TORCH_DISTRIBUTED_DEBUG was set to DETAIL, but
                                    GLOO is not available. Build with Gloo to
                                    create a wrapper process group in debug mode
                                    to aid collective desynchronization debugging."""
                        )
                    else:
                        backend_class = _create_process_group_wrapper(
                            wrapped_pg=backend_class,
                            store_prefix=group_name,
                            store=backend_prefix_store,
                            rank=group_rank,
                            world_size=group_size,
                            timeout=timeout,
                        )

            # register only a single backend when all get_device_backend_map values are the same
            if len(set(backend_config.get_device_backend_map().values())) == 1:
                for device in backend_config.get_device_backend_map().keys():
                    pg._register_backend(
                        torch.device(device), backend_type, backend_class
                    )

                # break out of outer loop to not create any more backends
                break

            pg._register_backend(torch.device(device), backend_type, backend_class)

        # set group_name and group_dsec to backend
        assert group_name is not None
        assert group_desc is not None
        pg._set_group_name(group_name)
        pg._set_group_desc(group_desc)

        if device_id and pg._get_backend(device_id).supports_splitting:
            eager_backend = pg._get_backend(device_id)
            eager_backend.eager_connect_single_device(device_id)

        # update global state
        _world.pg_map[pg] = (backend, prefix_store)
        _world.pg_names[pg] = group_name
        _register_process_group(group_name, pg)

        _world.pg_backend_config[pg] = str(backend_config)
        # "" is the default tag for user PGs
        if pg_tag in [None, ""]:
            pg_tag = f"ptd:{group_name}"
            _world.tags_to_pg.setdefault("", []).append(pg)
        else:
            pg_tag = f"user:{pg_tag}"

        _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
        _world.pg_to_tag[pg] = pg_tag
        return pg, prefix_store
