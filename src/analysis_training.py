"""
Analysis of models used for project:
"""
import glob
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_theme(style="darkgrid")
LABELMAP: str = r"data/training/annotations/label_map.pbtxt"  # to be used for model inference
RECORD_DIR: str = r"data/writeup_assets/records"  # dir that contains records for analysis
IMG_DIR: str = r"data/inference"


def build_table(data: [str, Dict]) -> None:
    """
    build table from given csv or dict
    :param data:
    :return:
    """
    if isinstance(data, str):  # from string path
        f = os.path.join(RECORD_DIR, data)
        df = pd.read_csv(f)

    elif isinstance(data, Dict):
        df = pd.DataFrame()
        for k in data.keys():
            df[k] = data[k]
    print(df.to_markdown())


def plot_img_grid(model: str, modelname: str) -> None:
    """
    plot a 2x2 grid of the predictions for a given model
    model: str path to image
    modelname: name for plot
    :return: None
    """
    # plot photos of predictions #
    fig, ax = plt.subplots(2, 2)
    imgfiles = glob.glob(model + "\*.png")
    # plot 4 images
    c = 0
    for i in range(2):
        for j in range(2):
            m_img = imgfiles[c]
            ax[i][j].axis("off")
            ax[i][j].set_title(f"Inference {c}")
            img = np.array(Image.open(m_img))
            ax[i][j].imshow(img)
            c += 1
    fig.suptitle(modelname)
    plt.tight_layout()
    plt.show()


def get_avg_pred_scores(model: str, min_len: int) -> Tuple[np.array, int]:
    """
    get average prediction scores for model
    :param model:
    :return:
    """
    # get scores for each prediction #
    scores = np.zeros(100)  # global for mean
    jsonfiles = glob.glob(model + "\*.json")
    for m_json in jsonfiles:
        with open(m_json, "r") as fp:
            # add score to total
            data = json.load(fp)
            # some models do not provide 100 boxes so we must trim to the minimum
            if len(scores) == len(data["detection_scores"]):
                scores += data["detection_scores"]
            else:
                scores = scores[: len(data["detection_scores"])]
                scores += data["detection_scores"]
                if min_len > len(data["detection_scores"]):
                    min_len = len(data["detection_scores"])
    # compute avg score
    scores /= len(jsonfiles)
    # store score for each model
    return scores, min_len


def analyze_run(dir: str) -> None:
    """
    analysis for training run
    dir: directory of run
    :return:
    """
    models = glob.glob(dir + "\*")
    score_dict = {}  # store the prediction scores for each model
    min_len = 100
    for m in models:
        # trim name for plots
        modelname = os.path.basename(m).split("_")[:2]
        modelname = "_".join(modelname)
        # plot given model
        plot_img_grid(m, modelname)  # plot the model
        # compute average prediction scores
        scores, min_len = get_avg_pred_scores(m, min_len)
        score_dict[modelname] = scores

    # truncate to match lowest num of scores
    for k in score_dict.keys():
        score_dict[k] = score_dict[k][:min_len]

    # plot scores and save plot.
    det_scores = pd.DataFrame.from_dict(score_dict)
    sns.lineplot(data=det_scores, palette="tab10", linewidth=2.5)
    plt.legend(loc=1)
    plt.show()

    # build table for scores
    for k in score_dict.keys():
        score_dict[k] = score_dict[k][:3]  # keep top three prediction values
    build_table(score_dict)


def analyze_out_of_box() -> None:
    """
    run analysis for out of box experiment
    :return:
    """
    oob_path = "data/writeup_assets/out-of-box-inference-final"
    analyze_run(oob_path)


def analyze_fine_tune() -> None:
    """
    run analysis for out of box experiment
    :return:
    """
    oob_path = "data/writeup_assets/fine-tune-inference"
    analyze_run(oob_path)


if __name__ == "__main__":
    # analyze_run(r"C:\Users\Noah Barrett\Desktop\School\fourth year (2020-2021)\CS 444\MLFinalProject\data\writeup_assets\fine-tune-inference")
    analyze_run(
        r"C:\Users\Noah Barrett\Desktop\School\fourth year (2020-2021)\CS 444\MLFinalProject\data\writeup_assets\out-of-box-inference-final"
    )
