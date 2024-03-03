"""
Inference.
ddd
Authors: Hongjie Fang.
"""
import os
import cv2
import yaml
import numpy as np
from mindspore import context, Tensor
import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tqdm import tqdm
import logging
import warnings
from time import perf_counter
from utils.logger import ColoredLogger  # Assuming this is custom and works with MindSpore
from utils.builder import ConfigBuilder  # Needs to be adapted for MindSpore

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Inferencer(object):
    """
    Inferencer for MindSpore.
    """
    def __init__(self, cfg_path=os.path.join('configs', 'inference.yaml'), with_info=False, **kwargs):
        """
        Initialization for MindSpore Inferencer.
        """
        warnings.filterwarnings("ignore")
        logging.setLoggerClass(ColoredLogger)
        self.logger = logging.getLogger(__name__)

        with open(cfg_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

        self.builder = ConfigBuilder(**cfg_params)  # Needs to be adapted for MindSpore
        self.with_info = with_info

        if self.with_info:
            self.logger.info('Building models ...')
        
        self.model = self.builder.get_model()  # Needs to be adapted for MindSpore

        self.cuda_id = self.builder.get_inference_cuda_id()
        context.set_context(device_id=self.cuda_id)

        if self.with_info:
            self.logger.info('Checking checkpoints ...')

        checkpoint_file = self.builder.get_inference_checkpoint_path()
        if os.path.isfile(checkpoint_file):
            param_dict = load_checkpoint(checkpoint_file)
            load_param_into_net(self.model, param_dict)
            if self.with_info:
                self.logger.info("Checkpoint {} loaded.".format(checkpoint_file))
        else:
            raise FileNotFoundError('No checkpoint.')

        self.image_size = self.builder.get_inference_image_size()
        self.depth_min, self.depth_max = self.builder.get_inference_depth_min_max()
        self.depth_norm = self.builder.get_inference_depth_norm()

    def inference(self, rgb, depth, target_size=(1280, 720)):
        """
        Inference method for MindSpore.
        """
        rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth = np.where(depth < self.depth_min, 0, depth)
        depth = np.where(depth > self.depth_max, 0, depth)
        depth[np.isnan(depth)] = 0
        depth = depth / self.depth_norm
        rgb = (rgb / 255.0).transpose(2, 0, 1)
        rgb = Tensor(rgb, dtype=mindspore.float32)
        depth = Tensor(depth, dtype=mindspore.float32)

        with mindspore.no_grad():
            time_start = perf_counter()
            depth_res = self.model(rgb, depth)
            time_end = perf_counter()

        if self.with_info:
            self.logger.info("Inference finished, time: {:.4f}s.".format(time_end - time_start))

        depth_res = depth_res.asnumpy().squeeze(0).squeeze(0)
        depth_res = depth_res * self.depth_norm
        depth_res = cv2.resize(depth_res, target_size, interpolation=cv2.INTER_NEAREST)
        return depth_res