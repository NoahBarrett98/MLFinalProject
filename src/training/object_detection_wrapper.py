"""
train/export models using tf object detection api
*** all models downloaded from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
and placed into appropriate folders manually ***

Models include:
'efficientDet_D0_512x512'
'CenterNet_Resnet50_V1_FPN_512x512'
'ssd_mobilenet_v2_320x320_coco17_tpu-8'
'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'

These functions just wrap
"""
import os

import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image


def train_model(model: str, tensorboard: bool = True, pipe_dir_run: str = "", additional_args: str = "") -> None:
    """
    show training process:
    Training was done on google colab.
    :param model: string name of model
    :param tensorboard: bool to toggle use of tensorboard.
    :param pipe_dir_run: use to specify the directory of which will be used for the training pipe
    :return:
    """
    if tensorboard:
        # launch tensorboard (api does all logging for us)
        logdir = f"./data/training/models/{model}/train/"
        os.system(f"tensorboard --logdir={logdir}")

    # fine-tune-archived model
    model_dir = f"data/training/models/{model}/"
    pipe_config = f"data/training/models/{model}/{pipe_dir_run}pipeline.config"
    os.system(
        f"python src/api_scripts/model_main_tf2.py --model_dir={model_dir} --pipeline_config_path={pipe_config}"
        + additional_args
    )


def export_model(model: str, session_dir: str = "") -> None:
    """
    export a model as a saved model that can be loaded using tensorflow
    using the tf object detection api
    *** if getting TypeError: Expected Operation, Variable, or Tensor, got ______
    must change line 140 in tf_utils.py to
    if not isinstance(x, str):
          raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))
    ***
    to export must have pipelne config and ckpt in same dir
    :param model: string name of model
    :param session_dir: dir that the train session exists in
    :return: None
    """
    # need to shorten the names of the exported model dirs because the temp files are too long for system
    shorten_model = [s[0] for s in model.split("_")]
    output_name = ""
    for s in shorten_model:
        output_name += s

    model_dir = f"data/training/models/{model}/{session_dir}/"
    pipe_config = f"data/training/models/{model}/{session_dir}/pipeline.config"
    output_dir = f"data/training/exported-models/{output_name}/"
    os.system(
        f"python src/api_scripts/exporter_main_v2.py --input_type image_tensor  --pipeline_config_path {pipe_config} --trained_checkpoint_dir {model_dir} --output_directory {output_dir}"
    )


def load_model(model: str, ckpt_num: int, ckpt_dir: str):
    """
    loads a model given a specific checkpoint
    """
    pipeline_config = f"{ckpt_dir}/pipeline.config"
    # ckpt_dir = f"./data/training/models/{model}/"

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(ckpt_dir, f"ckpt-{ckpt_num}")).expect_partial()

    return detection_model


@tf.function
def detect_fn(image, model):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections


def test_load(model: str, ckpt_num: int, ckpt_dir: str, img_dir: str):
    model = load_model(model, ckpt_num, ckpt_dir)

    # load image
    img = np.array(Image.open(img_dir))
    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, model)
    return detections
