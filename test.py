"""
Testing scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import numpy as np
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tqdm import tqdm
import argparse
import logging
import warnings
from utils.logger import ColoredLogger  # Assuming this is custom and works with MindSpore
from utils.builder import ConfigBuilder  # Needs to be adapted for MindSpore
from utils.functions import to_device  # Needs to be adapted for MindSpore
from time import perf_counter

# Set MindSpore context for GPU
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--cfg', '-c', 
    default=os.path.join('configs', 'inference.yaml'), 
    help='path to the configuration file', 
    type=str
)
args = parser.parse_args()
cfg_filename = args.cfg

with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)  # Needs to be adapted for MindSpore

logger.info('Building models ...')

model = builder.get_model()  # Needs to be adapted for MindSpore

# In MindSpore, device management is different
context.set_context(device_id=0)  # Assuming single GPU inference

logger.info('Building dataloaders ...')
test_dataloader = builder.get_dataloader(split='test')  # Needs to be adapted for MindSpore

logger.info('Checking checkpoints ...')
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.ckpt')
if os.path.isfile(checkpoint_file):
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(model, param_dict)
    # Additional code may be needed to handle 'start_epoch' or other checkpoint data
    logger.info("Checkpoint {} loaded.".format(checkpoint_file))
else:
    raise FileNotFoundError('No checkpoint.')

metrics = builder.get_metrics()  # Needs to be adapted for MindSpore

def test():
    logger.info('Start testing process.')
    model.set_train(False)  # Set model to evaluation mode
    metrics.clear()  # Assuming metrics is adapted for MindSpore
    running_time = []

    for data_dict in test_dataloader.create_dict_iterator():
        data_dict = to_device(data_dict)  # Assuming this is adapted for MindSpore
        time_start = perf_counter()

        res = model(Tensor(data_dict['rgb']), Tensor(data_dict['depth']))
        time_end = perf_counter()
        n, h, w = data_dict['depth'].shape
        data_dict['pred'] = res.view(n, h, w)

        _ = metrics.evaluate_batch(data_dict)  # Assuming metrics is adapted for MindSpore
        duration = time_end - time_start
        # Displaying the duration
        running_time.append(duration)

    avg_running_time = np.mean(running_time)
    logger.info('Finish testing process, average running time: {:.4f}s'.format(avg_running_time))
    metrics_result = metrics.get_results()  # Assuming metrics is adapted for MindSpore
    metrics.display_results()  # Assuming metrics is adapted for MindSpore
    return metrics_result

if __name__ == '__main__':
    test()
