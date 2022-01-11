# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
import os
import shutil
import tempfile
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from apex.transformer.pipeline_parallel.utils import get_num_microbatches
from deprecate.utils import void
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.loops.epoch.training_epoch_loop import TrainingEpochLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.utilities import _update_dataloader_iter
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataFetcher,
    DataLoaderIterDataFetcher,
    InterBatchParallelDataFetcher,
)
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _PATH
from pytorch_lightning.utilities.warnings import rank_zero_warn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.parts.utils_funcs import inject_model_parallel_rank
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


class NLPDDPPlugin(DDPPlugin):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.
    """

    accelerator = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        checkpoint_io: Optional[CheckpointIO] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, checkpoint_io, sync_batchnorm, **kwargs)

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        app_state = AppState()

        if app_state.model_parallel_size is not None:
            self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info(f"Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            # TODO: for megatron-lm self.model is a list
            self.pre_configure_ddp()
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=parallel_state.get_data_parallel_group(),
                **self._ddp_kwargs,
            )
            self._model.require_backward_grad_sync = False
            self._register_ddp_hooks()

        else:
            super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            if torch.distributed.is_initialized():
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=app_state.tensor_model_parallel_size,
                    pipeline_model_parallel_size_=app_state.pipeline_model_parallel_size,
                )

                # assert that fake tp and pp rank match after model parallel init
                assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
                assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

                app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
                app_state.data_parallel_group = parallel_state.get_data_parallel_group()
                app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
                app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
                app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH) -> None:
        """ PTL override to accomodate model parallel checkpoints """
        # TODO: move to CheckpointIO
        filepath = inject_model_parallel_rank(filepath)
        return super().save_checkpoint(checkpoint, filepath)

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        """ PTL override to accomodate model parallel checkpoints """
        # TODO: move to CheckpointIO
        torch.cuda.empty_cache()
        checkpoint_path = inject_model_parallel_rank(checkpoint_path)
        return self.checkpoint_io.load_checkpoint(checkpoint_path)

    def remove_checkpoint(self, filepath: _PATH) -> None:
        # PTL override to accomodate model parallel checkpoints
        filepath = self._inject_model_parallel_rank(filepath)
        logging.info(f'Removing checkpoint: {filepath}')
        return super().remove_checkpoint(filepath)

    def _inject_model_parallel_rank(self, filepath):
        return inject_model_parallel_rank(filepath)

    @property
    def should_rank_save_checkpoint(self) -> bool:
        # PTL override that determines if checkpoints should be saved based on rank
        # for model parallel we need data_parallel_rank==0
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            return app_state.data_parallel_rank == 0
        else:
            return super().should_rank_save_checkpoint

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPPlugin, self).distributed_sampler_kwargs


class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        super().__init__()
        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")

    def save_to(self, model, save_path: str):
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:

            dir_name = os.path.dirname(save_path)

            # first we save the weights for each model parallel rank
            if app_state.pipeline_model_parallel_size == 1:
                if app_state.data_parallel_rank == 0:
                    mp_model_weights = os.path.join(
                        dir_name, f'mp_rank_{app_state.tensor_model_parallel_rank:02d}_' + self.model_weights_ckpt
                    )

                    self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # create nemo file from folder with all mp_ranks checkpoints
                if app_state.tensor_model_parallel_rank == 0 and app_state.data_parallel_rank == 0:
                    with tempfile.TemporaryDirectory() as tmpdir:

                        # move weights to the tmpdir
                        for tp_rank in range(app_state.tensor_model_parallel_size):
                            os.makedirs(os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}'))
                            mp_model_weights = os.path.join(
                                dir_name, f'mp_rank_{tp_rank:02d}_' + self.model_weights_ckpt
                            )
                            shutil.move(
                                mp_model_weights,
                                os.path.join(tmpdir, f'mp_rank_{tp_rank:02d}', self.model_weights_ckpt),
                            )

                        # create config and artifacts in tmpdir
                        config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                        model.to_config_file(path2yaml_file=config_yaml)
                        if hasattr(model, 'artifacts') and model.artifacts is not None:
                            self._handle_artifacts(model, nemo_file_folder=tmpdir)
                            self._update_artifact_paths(model, path2yaml_file=config_yaml)

                        # create tar file
                        self._make_nemo_file_from_folder(save_path, tmpdir)
                else:
                    logging.info(
                        "Saving .nemo for pipeline parallel is not implemented yet. Please use a conversion script."
                    )

        else:
            return super().save_to(model, save_path)


class PipelineMixedPrecisionPlugin(NativeMixedPrecisionPlugin):
    """ Overrides PTL autocasting to not wrap training/val/test_step.
        We do this because we have the Apex fwd/bwd functions in training_step.
        This means .backward is being called in training_step so we do not want the whole
        step wrapped in autocast.

        We instead wrap the fwd_output_and_loss_func that is passed to the Apex fwd/bwd functions.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler=scaler)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Have the PTL context manager do nothing."""
        yield


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self, init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
        return retval

    def update(self, new_scale=None):
        """
        Updates the scale factor.

        If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
        to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
        the scale is multiplied by ``growth_factor`` to increase it.

        Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
        used directly, it's used to fill GradScaler's internal scale tensor. So if
        ``new_scale`` was a tensor, later in-place changes to that tensor will not further
        affect the scale GradScaler uses internally.)

        Args:
            new_scale (float or :class:`torch.cuda.FloatTensor`, optional, default=None):  New scale factor.

        .. warning::
            :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
            been invoked for all optimizers used this iteration.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
                    )
                    found_inf_combined += found_inf

            torch._amp_update_scale_(
                _scale,
                _growth_tracker,
                found_inf_combined,
                self._growth_factor,
                self._backoff_factor,
                self._growth_interval,
            )

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(torch.cuda.amp.grad_scaler._refresh_per_optimizer_state)


class NLPDataConnector(DataConnector):
    """ Override PTL DataConnector. Used to select custom data fetcher."""

    def __init__(
        self,
        trainer: "pl.Trainer",
        multiple_trainloader_mode: str = "max_size_cycle",
        train_data_fetcher: Optional[AbstractDataFetcher] = None,
        validate_data_fetcher: Optional[AbstractDataFetcher] = None,
        test_data_fetcher: Optional[AbstractDataFetcher] = None,
    ):

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")

        super().__init__(
            trainer,
            multiple_trainloader_mode=multiple_trainloader_mode,
            train_data_fetcher=train_data_fetcher,
            validate_data_fetcher=validate_data_fetcher,
            test_data_fetcher=test_data_fetcher,
        )

    def _select_data_fetcher(self) -> AbstractDataFetcher:
        if self.trainer.sanity_checking:
            return GlobalBatchDataFetcher()

        training_step_fx = getattr(self.trainer.lightning_module, "training_step")
        if self.trainer.training and is_param_in_hook_signature(training_step_fx, "dataloader_iter", explicit=True):
            rank_zero_warn(
                "Found `dataloader_iter` argument in the `training_step`. Note that the support for "
                "this signature is experimental and the behavior is subject to change."
            )
            return DataLoaderIterDataFetcher()

        elif self.trainer.training and os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1":
            # note: this is an experimental feature
            if not self.trainer.training_type_plugin.on_gpu:
                raise MisconfigurationException("Inter batch parallelism is available only when using Nvidia GPUs.")
            return InterBatchParallelDataFetcher()

        return GlobalBatchDataFetcher()


class GlobalBatchDataFetcher(DataFetcher):
    """ Overrides PTL DataFetcher. Used to fetch global batches."""

    def __init__(self, prefetch_batches: int = 0, store_on_device: bool = False) -> None:

        if not HAVE_APEX:
            logging.warning("Apex was not found. Using model parallel or megatron models will error out.")

        super().__init__(prefetch_batches=prefetch_batches, store_on_device=store_on_device)
        self.num_micro_batches = get_num_microbatches()

    def _fetch_next_batch(self):
        """ Fetches the next global batch which is a list of micro batches"""
        with self.apply_profiler(f"get_{self.stage}_batch"):
            with self.fetching_context():
                data = self.on_fetch_start()
                with self.apply_profiler(f"fetch_next_{self.stage}_batch"):
                    batch = [next(self.dataloader_iter) for _ in range(self.num_micro_batches)]
                self.fetched += 1
                self.on_fetch_end(batch, data)
