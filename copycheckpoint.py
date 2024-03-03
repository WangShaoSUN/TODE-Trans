"""
Testing scripts.

Authors: Hongjie Fang.
"""
import os
import yaml
import numpy as np
from mindspore import context, Tensor
from mindspore import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from tqdm import tqdm
import logging
import warnings
import argparse
from time import perf_counter
from utils.logger import ColoredLogger  # Assuming compatibility with MindSpore
from utils.builder import ConfigBuilder  # Needs adaptation for MindSpore
from utils.functions import to_device  # Needs adaptation for MindSpore

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
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

builder = ConfigBuilder(**cfg_params)  # Needs adaptation for MindSpore

logger.info('Building models ...')

model = builder.get_model()  # Needs adaptation for MindSpore

context.set_context(device_id=0)  # Assuming single GPU inference

logger.info('Building dataloaders ...')
test_dataloader = builder.get_dataloader(split='test')  # Needs adaptation for MindSpore

logger.info('Checking checkpoints ...')
stats_dir = builder.get_stats_dir()
checkpoint_file = os.path.join(stats_dir, 'checkpoint.ckpt')
if os.path.isfile(checkpoint_file):
    param_dict = load_checkpoint(checkpoint_file)
    load_param_into_net(model, param_dict)
    logger.info("Checkpoint {} loaded.".format(checkpoint_file))
else:
    raise FileNotFoundError('No checkpoint.')

metrics = builder.get_metrics()  # Needs adaptation for MindSpore

def test():
    logger.info('Start testing process.')
    model.set_train(False)
    metrics.clear()  # Assuming adaptation for MindSpore
    running_time = []

    for data_dict in test_dataloader.create_dict_iterator():  # Assuming adaptation for MindSpore
        data_dict = to_device(data_dict)  # Assuming adaptation for MindSpore
        with mindspore.no_grad():
            time_start = perf_counter()
            res = model(Tensor(data_dict['rgb'], mstype.float32), Tensor(data_dict['depth'], mstype.float32))
            time_end = perf_counter()
            n, h, w = data_dict['depth'].shape
            data_dict['pred'] = res.view(n, h, w).asnumpy()  # Adjust for MindSpore tensor handling

            _ = metrics.evaluate_batch(data_dict)  # Assuming adaptation for MindSpore
            duration = time_end - time_start
            pbar.set_description('Time: {:.4f}s'.format(duration))
            running_time.append(duration)

    avg_running_time = np.mean(running_time)
    logger.info('Finish testing process, average running time: {:.4f}s'.format(avg_running_time))
    metrics_result = metrics.get_results()  # Assuming adaptation for MindSpore
    metrics.display_results()  # Assuming adaptation for MindSpore
    return metrics_result

if __name__ == '__main__':
    test()
