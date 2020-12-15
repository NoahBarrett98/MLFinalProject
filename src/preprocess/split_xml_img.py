import argparse
import glob
import os
import shutil

"""
Author: Noah Barrett
split files into xml and images dirs
******unsafe and quickly thrown together*****
"""
# Initiate argument parser
parser = argparse.ArgumentParser(description="Split Labelimg generated dir into labels, imgs")
parser.add_argument("-dir", "--dir", help="Path to the folder where the data and .xml files are stored.", type=str)


args = parser.parse_args()

DIR = args.dir


def sort_imgs(dir: str, img_ext: str = ".jpg") -> None:
    """
    sorts imgs into two dirs: xml and image
    :param dir: str path for directory containing images
    :param img_ext: extension of image
    :return: none
    """
    xml = glob.glob(dir + "\*.xml")
    img = [os.path.join(os.path.dirname(x), os.path.basename(x).split(".")[0] + img_ext) for x in xml]
    xml_path = os.path.join(dir, "xml")
    img_path = os.path.join(dir, "images")
    os.mkdir(xml_path)
    os.mkdir(img_path)
    for x, i in zip(xml, img):
        shutil.move(x, xml_path)
        shutil.move(i, img_path)


if __name__ == "__main__":
    sort_imgs(DIR)
