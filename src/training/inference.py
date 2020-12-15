"""
script to perform inference on trained models:
credit to https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # need to supress GPU due to high memory usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import glob

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logging (2)
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image

warnings.filterwarnings("ignore")  # Suppress Matplotlib warnings
LABELMAP: str = r"data/training/annotations/label_map.pbtxt"  # to be used for model inference
IMG_DIR: str = r"data/inference/*"


def run_inference(
    PATH_TO_CFG: str, PATH_TO_CKPT: str, PATH_TO_LABELS: str, IMAGE_PATHS: str, OUTPUT_DIR: str, CKPT_NUM: str
) -> None:
    """
    This function is adapted *very closely* from:
    https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/plot_object_detection_checkpoint.html
    :param PATH_TO_CFG:
    :param PATH_TO_CKPT:
    :param PATH_TO_LABELS:
    :param IMAGE_PATHS:
    :param OUTPUT_DIR:
    :param CKPT_NUM:
    :return:
    """
    IMAGE_PATHS = glob.glob(IMAGE_PATHS)
    print("Loading model... ", end="")
    start_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, f"ckpt-{CKPT_NUM}")).expect_partial()

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Done! Took {} seconds".format(elapsed_time))

    # labelmap
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    for i, image_path in enumerate(IMAGE_PATHS):

        print("Running inference for {}... ".format(image_path), end="")
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop("num_detections"))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections["num_detections"] = num_detections

        # detection_classes should be ints.
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        try:
            json_det = detections.copy()
            # "jsonify" outputs
            for k in json_det.keys():
                if isinstance(json_det[k], np.ndarray):
                    json_det[k] = detections[k].tolist()
            with open(f"{OUTPUT_DIR}/inference{i}.json", "w") as fp:
                json.dump(json_det, fp)
        except FileNotFoundError:
            os.mkdir(OUTPUT_DIR)
            # "jsonify" outputs
            for k in json_det.keys():
                if isinstance(json_det[k], np.ndarray):
                    json_det[k] = json_det[k].tolist()
            with open(f"{OUTPUT_DIR}/inference{i}.json", "w") as fp:
                json.dump(json_det, fp)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + label_id_offset,
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=3,
            min_score_thresh=0.30,
            agnostic_mode=False,
        )

        plt.figure()
        plt.imshow(image_np_with_detections)
        print("Done")
        if not os.path.isdir(f"{OUTPUT_DIR}/inference"):
            os.mkdir(f"{OUTPUT_DIR}/inference")
        plt.imsave(f"{OUTPUT_DIR}/inference{i}.png", image_np_with_detections)


def inference_for_out_of_box() -> None:
    """
    performs an inference run of the models that were trained for the out of box experiment
    :return:
    """
    model_dict = {
        "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8": 9,
        "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8": 2,
        "centernet_resnet50_v1_fpn_512x512_coco17_tpu-8": 2,
        "efficientdet_d0_coco17_tpu-32": 3,
        "ssd_mobilenet_v2_320x320_coco17_tpu-8": 3,
    }
    for model in model_dict.keys():
        output_dir = f"data/writeup_assets/out-of-box-inference/{model}"
        config = f"data/training/models/{model}/out-of-box/pipeline.config"
        ckpt = f"data/training/models/{model}/out-of-box/ckpt/"
        run_inference(
            PATH_TO_CFG=config,
            PATH_TO_CKPT=ckpt,
            PATH_TO_LABELS=LABELMAP,
            IMAGE_PATHS=IMG_DIR,
            OUTPUT_DIR=output_dir,
            CKPT_NUM=model_dict[model],
        )


def inference_for_fine_tune() -> None:
    """
    performs an inference run of the models that were trained for the fine tune experiment
    :return:
    """
    model_dict = {
        "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8": 26,
        "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8": 20,
    }
    for model in model_dict.keys():
        output_dir = f"data/writeup_assets/out-of-box-inference/{model}"
        config = f"data/training/models/{model}/fine-tune/pipeline.config"
        ckpt = f"data/training/models/{model}/fine-tune/ckpt/"
        run_inference(
            PATH_TO_CFG=config,
            PATH_TO_CKPT=ckpt,
            PATH_TO_LABELS=LABELMAP,
            IMAGE_PATHS=IMG_DIR,
            OUTPUT_DIR=output_dir,
            CKPT_NUM=model_dict[model],
        )
