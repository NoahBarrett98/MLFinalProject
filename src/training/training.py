"""
script used for training on colab/local device

"""
import sys

# for google colab #
sys.path.append("/content/gdrive/My Drive/MLFinalProject/")
from src.training.object_detection_wrapper import *


def out_of_box_train() -> None:
    """
    Train for number of steps found to produce best loss,
    anything below 1000 steps rounded up to 1000,
    This will not produce same results due to shuffling, but
    will get in general ball park of original results
    ** Numsteps determined from first out of box train **
    saves checkpoints to the directory the model exists in.
    :return: None
    """
    models = [
        ("ssd_mobilenet_v2_320x320_coco17_tpu-8", 1000),
        ("efficientdet_d0_coco17_tpu-32", 1000),
        ("centernet_resnet50_v1_fpn_512x512_coco17_tpu-8", 1600),
        ("faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8", 1000),
        ("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8", 8100),
    ]
    for model, num_steps in models:
        train_model(
            model=model,
            tensorboard=False,
            pipe_dir_run="out-of-box-tensorboard-loss/",  # run using pipeline in out-of-box-tensorboard-loss directory
            additional_args=f" --num_train_steps={num_steps}",
        )


def export_out_of_box_models():
    """
    export models trained with out_of_box_train()
    :return:
    """
    models = [
        "ssd_mobilenet_v2_320x320_coco17_tpu-8",
        "efficientdet_d0_coco17_tpu-32",
        "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8",
        "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8",
        "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8",
    ]
    for model in models:
        export_model(
            model=model, session_dir="out-of-box-tensorboard-loss"  # run using out-of-box-tensorboard-loss directory
        )


def fine_tune_train() -> None:
    """
    continue training best performing models
    saves checkpoints to the directory the model exists in.

    *** Note that all configurations must be made to the models
    configs files for "fine tuning" the exact config file for this
    can be found in the fine-tune-archived dir***

    -> numsteps = max on config file
    :return: None
    """
    models = ["faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8", "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"]
    for model in models:
        train_model(model=model, tensorboard=False, pipe_dir_run="fine-tune/")
