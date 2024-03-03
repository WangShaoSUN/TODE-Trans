"""
Training scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import numpy as np
from mindspore import context
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_rank
from mindspore.common.initializer import initializer
from tqdm import tqdm
import argparse
import logging
import warnings
# Assuming this is custom and works with MindSpore
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder  # Needs to be adapted for MindSpore
from utils.constants import LOSS_INF  # Assuming this is a constant value
# Needs to be adapted for MindSpore
from utils.functions import display_results, to_device
from time import perf_counter
from mindspore import Tensor
from mindspore import ops
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.common import dtype as mstype
import numpy as np
from mindspore import save_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


# Set MindSpore context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument(
    '--cfg', '-c',
    default=os.path.join('configs', 'default.yaml'),
    help='path to the configuration file',
    type=str
)
args = parser.parse_args()
cfg_filename = args.cfg

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)  # Needs to be adapted for MindSpore

tensorboard_log = builder.get_tensorboard_log()

logger.info('Building models ...')

model = builder.get_model()  # Needs to be adapted for MindSpore

# Multi-GPU setup in MindSpore is different from PyTorch
# Assuming builder.multigpu() checks if multiple GPUs should be used
if builder.multigpu():
    init()
    model.set_train()
else:
    print('No GPUs, cannot initialize multigpu training.')

logger.info('Building dataloaders ...')
# Dataloader building process might be different in MindSpore
train_dataset = builder.get_dataloader(
    split='train')  # Needs to be adapted for MindSpore
test_dataset = builder.get_dataloader(
    split='test')  # Needs to be adapted for MindSpore
test_real_dataloader = builder.get_dataloader(dataset_params={"test": {"type": "cleargrasp-syn", "data_dir": "cleargrasp", "image_size": (320, 240),
                                                                       "use_augmentation": False, "depth_min": 0.0, "depth_max": 10.0,  "depth_norm": 1.0}}, split='test')  # Needs to be adapted for MindSpore

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = builder.get_max_epoch()
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.ckpt')

if os.path.isfile(checkpoint_file):
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(model, param_dict)
    # Additional steps may be needed to set start_epoch, metrics, and loss from checkpoint
    logger.info("Checkpoint {} (epoch {}) loaded.".format(
        checkpoint_file, start_epoch))

logger.info('Building optimizer and learning rate schedulers ...')
resume = (start_epoch > 0)
# Needs to be adapted for MindSpore
optimizer = builder.get_optimizer(model, resume=resume, resume_lr=1e-4)
lr_scheduler = builder.get_lr_scheduler(optimizer, resume=resume, resume_epoch=(
    start_epoch - 1 if resume else None))  # Needs to be adapted for MindSpore

criterion = builder.get_criterion()  # Needs to be adapted for MindSpore
metrics = builder.get_metrics()  # Needs to be adapted for MindSpore


def train_one_epoch(epoch, model, train_dataset, criterion, optimizer):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))

    model.set_train()
    losses = []
    with tqdm(train_dataset.create_dict_iterator()) as pbar:
        for data_dict in pbar:
            data_dict = data_dict['data']
            res = model(data_dict['rgb'], data_dict['depth'])
            n, h, w = data_dict['depth'].shape
            data_dict['pred'] = res.view(n, h, w)

            loss_dict = criterion(data_dict)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'smooth' in loss_dict.keys():
                pbar.set_description('Epoch {}, loss: {:.8f}, smooth loss: {:.8f}'.format(epoch + 1, loss.item(), loss_dict['smooth'].item()))
            else:
                pbar.set_description('Epoch {}, loss: {:.8f}'.format(epoch + 1, loss.item()))

            losses.append(loss.mean().asnumpy())

    mean_loss = np.mean(losses)
    logger.info('Finish training process in epoch {}, mean training loss: {:.8f}'.format(
        epoch + 1, mean_loss))
    return mean_loss


def test_one_epoch(epoch, model, dataloader, criterion, metrics):
    logger.info('Start testing process in epoch {}.'.format(epoch + 1))
    model.set_train(False)
    metrics.clear()
    losses = []
    running_time = []

    for data_dict in dataloader.create_dict_iterator():
        # Assuming this is adapted for MindSpore
        data_dict = to_device(data_dict)
        start_time = perf_counter()
        res = model(data_dict['rgb'], data_dict['depth'])
        n, h, w = data_dict['depth'].shape
        data_dict['pred'] = res.view(n, h, w)

        loss_dict = criterion(data_dict)
        loss = loss_dict['loss']

        # Assuming metrics is adapted for MindSpore
        _ = metrics.evaluate_batch(data_dict)
        end_time = perf_counter()

        if 'smooth' in loss_dict.keys():
            # Displaying the loss information
            pass

        losses.append(loss.asnumpy())
        running_time.append(end_time - start_time)

    mean_loss = np.mean(losses)
    avg_running_time = np.mean(running_time)
    logger.info('Finish testing process in epoch {}, mean testing loss: {:.8f}, average running time: {:.4f}s'.format(
        epoch + 1, mean_loss, avg_running_time))
    # Assuming metrics is adapted for MindSpore
    metrics_result = metrics.get_results()
    metrics.display_results()  # Assuming metrics is adapted for MindSpore
    return mean_loss, metrics_result


def train(start_epoch):
    # Initial setup remains mostly the same
    if start_epoch != 0:
        min_loss = checkpoint_loss
        min_loss_epoch = start_epoch
        display_results(checkpoint_metrics, logger)
    else:
        min_loss = LOSS_INF
        min_loss_epoch = None

    # Configure the checkpoint saving
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=max_epoch, keep_checkpoint_max=5)
    ckpt_callback = ModelCheckpoint(
        prefix="checkpoint", directory=stats_dir, config=ckpt_config)


    for epoch in range(start_epoch, max_epoch):
        logger.info('--> Epoch {}/{}'.format(epoch + 1, max_epoch))
        train_loss = train_one_epoch(epoch, model, train_dataset, criterion, optimizer)
        real_test_loss, metrics_result = test_one_epoch(test_dataloader, epoch)

        syn_test_loss, _ = test_one_epoch(test_real_dataloader, epoch)

        
        tensorboard_log.add_scalar("train_mean_loss", train_loss, epoch)
        tensorboard_log.add_scalar("real_mean_loss", real_test_loss, epoch)
        tensorboard_log.add_scalar("syn_mean_loss", syn_test_loss, epoch)

        # Save checkpoint for every epoch
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss': real_test_loss,
            'metrics': metrics_result
        }
        save_checkpoint(save_dict, os.path.join(
            stats_dir, 'checkpoint-epoch{}.ckpt'.format(epoch)), append_dict=False)

        # Update minimum loss
        if real_test_loss < min_loss:
            min_loss = real_test_loss
            min_loss_epoch = epoch + 1
            save_checkpoint(save_dict, os.path.join(
                stats_dir, 'checkpoint.ckpt'), append_dict=False)

    logger.info('Training Finished. Min testing loss: {:.6f}, in epoch {}'.format(
        min_loss, min_loss_epoch))
    tensorboard_log.close()


if __name__ == '__main__':
    train(start_epoch=start_epoch)
