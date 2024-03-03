"""
Configuration builder.

Authors: Hongjie Fang.
"""
from mindspore import dataset as ds
import os
import logging
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore.common import set_seed
from utils.logger import ColoredLogger  # Assuming compatibility with MindSpore
# MindSpore equivalent for TensorBoard
from mindspore.train.callback import SummaryCollector
import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import Compose, TypeCast
from mindspore.train.callback import SummaryCollector
import mindspore.nn as mnn
from mindspore.train.metrics import Metric
import numpy as np
logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


class ConfigBuilder(object):
    """
    Configuration Builder for MindSpore.
    """

    def __init__(self, **params):
        """
        Set the default configuration for the configuration builder in MindSpore context.

        Parameters
        ----------

        params: the configuration parameters.
        """
        super(ConfigBuilder, self).__init__()
        self.params = params
        self.model_params = params.get('model', {})
        self.optimizer_params = params.get('optimizer', {})
        self.lr_scheduler_params = params.get('lr_scheduler', {})
        self.dataset_params = params.get('dataset', {'data_dir': 'data'})
        self.dataloader_params = params.get('dataloader', {})
        self.trainer_params = params.get('trainer', {})
        self.metrics_params = params.get('metrics', {})
        self.stats_params = params.get('stats', {})
        self.inference_params = params.get('inference', {})
        self.tb_log = params.get('tb_log', {})

    def get_model(self, model_params=None):
        """
        Get the model from configuration in MindSpore.

        Parameters
        ----------

        model_params: dict, optional, default: None.

        Returns
        -------

        A model, which is usually a mindspore.nn.Cell object.
        """
        # Example: Importing a custom model from MindSpore
        from models.Tode import Tode
        if model_params is None:
            model_params = self.model_params
        type = model_params.get('type', 'custom_type')
        params = model_params.get('params', {})
        if type == 'tode':
            model = Tode(**params)
        else:
            raise NotImplementedError('Invalid model type.')
        return model

    def get_optimizer(self, model, optimizer_params=None, resume=False, resume_lr=None):
        """
        Get the optimizer for MindSpore.

        Parameters
        ----------
        model: a mindspore.nn.Cell object, the model.
        optimizer_params: dict, optional, default: None.
        resume: bool, optional, default: False, whether to resume training.
        resume_lr: float, optional, default: None, the resume learning rate.

        Returns
        -------
        An optimizer for the given model.
        """
        if optimizer_params is None:
            optimizer_params = self.optimizer_params
        type = optimizer_params.get('type', 'Adam')
        params = optimizer_params.get('params', {})
        if resume:
            params.update(learning_rate=resume_lr)
        if type == 'SGD':
            optimizer = nn.SGD(params=model.trainable_params(), **params)
        elif type == 'Adam':
            optimizer = nn.Adam(params=model.trainable_params(), **params)
        elif type == 'AdamW':
            optimizer = nn.AdamWeightDecay(
                params=model.trainable_params(), **params)
        else:
            raise NotImplementedError('Invalid optimizer type for MindSpore.')
        return optimizer

    def get_lr_scheduler(self, optimizer, lr_scheduler_params=None, resume=False, resume_epoch=None):
        """
        Get the learning rate scheduler for MindSpore.

        Parameters
        ----------
        optimizer: an optimizer for MindSpore.
        lr_scheduler_params: dict, optional.
        resume: bool, optional, whether to resume training.
        resume_epoch: int, optional, the epoch of the checkpoint.

        Returns
        -------
        A learning rate scheduler for the given optimizer.
        """
        return '' 

    def get_dataset(self, dataset_params=None, split='train'):
        """
        Get the dataset from configuration.

        Parameters
        ----------

        dataset_params: dict, optional, default: None. If dataset_params is provided, then use the parameters specified in the dataset_params to build the dataset. Otherwise, the dataset parameters in the self.params will be used to build the dataset;

        split: str in ['train', 'test'], optional, default: 'train', the splitted dataset.

        Returns
        -------

        A torch.utils.data.Dataset item.
        """
        from datasets.transcg import TransCG
        from datasets.cleargrasp import ClearGraspRealWorld, ClearGraspSynthetic
        from datasets.omniverse_object import OmniverseObject
        from datasets.transparent_object import TransparentObject
        from datasets.mixed_dataset import MixedDataset
        if dataset_params is None:
            dataset_params = self.dataset_params
        dataset_params = dataset_params.get(split, {'type': 'transcg'})
        if type(dataset_params) == dict:
            dataset_type = str.lower(dataset_params.get('type', 'transcg'))
            if dataset_type == 'transcg':
                dataset = TransCG(split=split, **dataset_params)
            elif dataset_type == 'cleargrasp-real':
                dataset = ClearGraspRealWorld(split=split, **dataset_params)
            elif dataset_type == 'cleargrasp-syn':
                dataset = ClearGraspSynthetic(split=split, **dataset_params)
            elif dataset_type == 'omniverse':
                dataset = OmniverseObject(split=split, **dataset_params)
            elif dataset_type == 'transparent-object':
                dataset = TransparentObject(split=split, **dataset_params)
            elif dataset_type == 'mixed-object':
                dataset = MixedDataset(split=split, **dataset_params)
            else:
                raise NotImplementedError(
                    'Invalid dataset type: {}.'.format(dataset_type))
            logger.info('Load {} dataset as {}ing set with {} samples.'.format(
                dataset_type, split, len(dataset)))
        elif type(dataset_params) == list:
            dataset_types = []
            dataset_list = []
            for single_dataset_params in dataset_params:
                dataset_type = str.lower(
                    single_dataset_params.get('type', 'transcg'))
                if dataset_type in dataset_types:
                    raise AttributeError('Duplicate dataset found.')
                else:
                    dataset_types.append(dataset_type)
                if dataset_type == 'transcg':
                    dataset = TransCG(split=split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-real':
                    dataset = ClearGraspRealWorld(
                        split=split, **single_dataset_params)
                elif dataset_type == 'cleargrasp-syn':
                    dataset = ClearGraspSynthetic(
                        split=split, **single_dataset_params)
                elif dataset_type == 'omniverse':
                    dataset = OmniverseObject(
                        split=split, **single_dataset_params)
                elif dataset_type == 'transparent-object':
                    dataset = TransparentObject(
                        split=split, **single_dataset_params)
                else:
                    raise NotImplementedError(
                        'Invalid dataset type: {}.'.format(dataset_type))
                dataset_list.append(dataset)
                logger.info('Load {} dataset as {}ing set with {} samples.'.format(
                    dataset_type, split, len(dataset)))
            dataset = ds.zip(dataset_list)
        else:
            raise AttributeError('Invalid dataset format.')
        return dataset

    def get_dataloader(self, dataset_params=None, split='train', batch_size=None, dataloader_params=None):
        """
        Get the dataloader for MindSpore.

        Parameters
        ----------
        dataset_params: dict, optional.
        split: str, 'train' or 'test'.
        batch_size: int, optional.
        dataloader_params: dict, optional.

        Returns
        -------
        A MindSpore dataloader.
        """
        if batch_size is None:
            if split == 'train':
                batch_size = self.trainer_params.get('batch_size', 32)
            else:
                batch_size = self.trainer_params.get('test_batch_size', 1)

        if dataloader_params is None:
            dataloader_params = self.dataloader_params

        dataset = self.get_dataset(dataset_params, split)
        # MindSpore's GeneratorDataset or other dataset types can be used
        # Transformations and operations on the dataset can be defined here

        return GeneratorDataset(dataset, column_names=["data"], shuffle=(split == 'train'))

    def get_max_epoch(self, trainer_params=None):
        """
        Get the max epoch from configuration for MindSpore.
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('max_epoch', 40)

    def get_stats_dir(self, stats_params=None):
        """
        Get the statistics directory from configuration for MindSpore.
        """
        if stats_params is None:
            stats_params = self.stats_params
        stats_dir = stats_params.get('stats_dir', 'stats')
        stats_exper = stats_params.get('stats_exper', 'default')
        stats_res_dir = os.path.join(stats_dir, stats_exper)
        if not os.path.exists(stats_res_dir):
            os.makedirs(stats_res_dir)
        return stats_res_dir

    def get_tensorboard_log(self, tb_log=None):
        """
        Get the TensorBoard log directory for MindSpore.
        MindSpore uses SummaryCollector instead of SummaryWriter.
        """
        if tb_log is None:
            tb_log = self.tb_log
        tb_dir = tb_log.get('stats_dir', 'stats')
        tb_exper = tb_log.get('stats_exper', 'default')
        tb_res_dir = os.path.join(tb_dir, tb_exper)
        if not os.path.exists(tb_res_dir):
            os.makedirs(tb_res_dir)
        collector = SummaryCollector(summary_dir=tb_res_dir)
        return collector

    def multigpu(self, trainer_params=None):
        """
        Get the multigpu settings from configuration for MindSpore.
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('multigpu', False)

    def get_resume_lr(self, trainer_params=None):
        """
        Get the resume learning rate from configuration for MindSpore.
        """
        if trainer_params is None:
            trainer_params = self.trainer_params
        return trainer_params.get('resume_lr', 0.001)

    def get_criterion(self, criterion_params=None):
        """
        Get the criterion settings for MindSpore from configuration.
        """
        if criterion_params is None:
            criterion_params = self.trainer_params.get('criterion', {})
        loss_type = criterion_params.get('type', 'custom_masked_mse_loss')

        # Map the loss types to MindSpore's loss functions
        if criterion_params is None:
            criterion_params = self.trainer_params.get('criterion', {})
        loss_type = criterion_params.get('type', 'custom_masked_mse_loss')
        from utils.criterion import Criterion
        criterion = Criterion(**criterion_params)
        return criterion

    def get_metrics(self, metrics_params=None):
        """
        Get the metrics settings for MindSpore from configuration.
        """
        if metrics_params is None:
            metrics_params = self.metrics_params
        metrics_list = metrics_params.get('types', ['MSE', 'MaskedMSE', 'RMSE', 'MaskedRMSE', 'REL', 'MaskedREL', 'MAE', 'MaskedMAE', 'Threshold@1.05', 'MaskedThreshold@1.05', 'Threshold@1.10', 'MaskedThreshold@1.10', 'Threshold@1.25', 'MaskedThreshold@1.25'])
        from utils.metrics import MetricsRecorder
        metrics = MetricsRecorder(metrics_list = metrics_list, **metrics_params)
        return metrics

    def get_inference_image_size(self, inference_params=None):
        """
        Get the inference image size from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference image size. Otherwise, the inference parameters in the self.params will be used to get the inference image size.

        Returns
        -------

        Tuple of (int, int), the image size.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('image_size', (320, 240))

    def get_inference_checkpoint_path(self, inference_params=None):
        """
        Get the inference checkpoint path from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference checkpoint path. Otherwise, the inference parameters in the self.params will be used to get the inference checkpoint path.

        Returns
        -------

        str, the checkpoint path.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('checkpoint_path', os.path.join('checkpoints', 'checkpoint.tar'))

    def get_inference_cuda_id(self, inference_params=None):
        """
        Get the inference CUDA ID from inference configuration.

        Parameters
        ----------

        inference_params: dict, optional, default: None. If inference_params is provided, then use the parameters specified in the inference_params to get the inference CUDA ID. Otherwise, the inference parameters in the self.params will be used to get the inference CUDA ID.

        Returns
        -------

        int, the CUDA ID.
        """
        if inference_params is None:
            inference_params = self.inference_params
        return inference_params.get('cuda_id', 0)

    def get_inference_depth_min_max(self, inference_params=None):
        """
        Get the min and max depth from inference configuration for MindSpore.
        """
        if inference_params is None:
            inference_params = self.inference_params
        depth_min = inference_params.get('depth_min', 0.3)
        depth_max = inference_params.get('depth_max', 1.5)
        return depth_min, depth_max

    def get_inference_depth_norm(self, inference_params=None):
        """
        Get the depth normalization coefficient from inference configuration for MindSpore.
        """
        if inference_params is None:
            inference_params = self.inference_params
        depth_norm = inference_params.get('depth_norm', 1.0)
        return depth_norm
