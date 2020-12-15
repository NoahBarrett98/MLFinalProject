import glob
import xml.etree.ElementTree as ET
from typing import List, Tuple

import matplotlib
import numpy as np
from numba import jit
from scipy.stats import mannwhitneyu

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
from sklearn.metrics import roc_auc_score

# original size, reshaped for different models
IMG_SIZE: Tuple[int, int] = 4032, 3024
RESIZE_FACTOR: int = 8

def get_bounding_boxes(xml_path) -> List[int]:
    """
    extract bounding boxes from xml file
    :param xml_path: str path to xml file
    :return:
    """
    # parse xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for box in root.findall("./object/bndbox"):
        bboxes.append([int(l.text) for l in list(box.iter())[1:]])
    return bboxes

# @jit
def gen_np_mask(bboxes: List[int], img_size: Tuple[int, int] =IMG_SIZE) -> np.array:
    """
    generate a mask for the image given a bounding box
    :param bboxes:
    :param img_size:
    :return:
    """
    mask = np.zeros(img_size)
    for xmin, ymin, xmax, ymax in bboxes:
        mask[xmin:xmax,ymin:ymax] = 1.0
    mask = rotate(mask, angle=-90)
    return mask

# @jit
def load_mask(xml_path: str) -> np.array:
    bboxes = get_bounding_boxes(xml_path)
    mask = gen_np_mask(bboxes)
    return mask

# @jit
def load_masks(data_dir: str) -> np.array:
    """
    loads masks from a dir of xml files
    :param data_dir:
    :return:
    """
    xmls = glob.glob(data_dir+ "\*.xml")
    mask_np = []
    for xml in xmls:
        mask = load_mask(xml)
        mask = resize_sample(mask)
        mask_np.append(mask)

    return np.array(mask_np)

# @jit
def pre_process_img(img_path: str) -> np.array:
    # open as gray scale
    img = Image.open(img_path)
    # convert to greyscale
    grayscale_image = np.dot(np.array(img)[..., :3], [0.2989, 0.5870, 0.1140])
    # plt.imshow(grayscale_image, cmap='gray')
    # plt.show()
    return grayscale_image

# @jit
def load_images(data_dir: str) -> np.array:
    """
    given a dir load entire dir as string
    :param data_dir:
    :return:
    """
    images = glob.glob(data_dir+"\*.jpg")
    imgs_np = []
    for img in images:
        img = pre_process_img(img)
        img = resize_sample(img)
        imgs_np.append(img)
    return np.array(imgs_np)


# @jit
def pixel_wise_auc(data_dir: str) -> None:
    """
    compute auc between label and image converted to grayscale
    :param mask: np.array mask
    :param img: np.array true image
    :return:
    """
    size = int(IMG_SIZE[0]/RESIZE_FACTOR), int(IMG_SIZE[1]/RESIZE_FACTOR)
    PW_AUC = np.zeros(size)
    # compute auc for entire image and mask
    imgs = load_images(data_dir)
    masks = load_masks(data_dir)
    # plt.imshow(masks[0])
    # plt.show()
    for i in range(len(PW_AUC)):
        for j in range(len(PW_AUC[0])):
            x = [img[i, j] for img in imgs]
            y = [mask[i, j] for mask in masks]
            # compute pixel wise auc
            auc = mannwhitneyu(x, y).statistic / (len(x) * len(y))
            PW_AUC[i, j] = auc
    plt.imshow(PW_AUC, cmap="seismic")
    plt.show()
    plt.imsave("data/writeup_assets/AUC_PW.png", PW_AUC)


# @jit
def resize_sample(arr: np.array, resize_factor: int = RESIZE_FACTOR):
    newsize = int(arr.shape[0] / resize_factor), int(arr.shape[1] / resize_factor)
    arr = Image.fromarray(arr)
    arr = np.array(arr.resize(newsize))
    return arr



@jit
def split_n_blocks(arr: np.array, nblocks: int) -> np.array:
    """
    *** Adapted from https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    Takes np array and divides up into sub-blocks
    """
    h, w = arr.shape
    nrows, ncols = int(h / np.sqrt(nblocks)), int(w / np.sqrt(nblocks))

    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

@jit
def area_wise_auc(mask: np.array, img: np.array):
    assert mask.shape == img.shape
    s_img = split_n_blocks(img, 9)
    s_mask = split_n_blocks(mask, 9)
    # compute averages for each block
    avg_img, avg_mask = [], []
    for i, (img, mask) in enumerate(zip(s_img, s_mask)):
        avg_img.append(np.mean(img))
        avg_mask.append(np.mean(mask))

    # convert to np arrays
    avg_img, avg_mask = np.array(avg_img), np.array(avg_mask)
    aucs = np.zeros(9)
    for i, (a_i, a_m) in enumerate(zip(avg_img, avg_mask)):
        data = np.concatenate([[a_i], [a_m]])
        labels = np.array([0] + [1])
        aucs[i] = roc_auc_score(y_true=labels, y_score=data)

    aucs = aucs.reshape((3,3))
    plt.imshow(aucs)
    plt.show()

