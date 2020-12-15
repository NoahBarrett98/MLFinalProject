"""
Author: Noah Barrett
Script to generate dataset

This script can be used to generate a dataset given any backdrop and entity object
"""
import random

from PIL import Image

ROBO: str = r".\data\training\images\assets\robo.jpg"
BACKDROP: str = r".\data\training\images\assets\backdrop.jpg"


def random_perturb(num: int, instance: str = ROBO, backdrop: str = BACKDROP, num_instances: int = 3) -> None:
    """
    Randomly perturb object across back drop N times
    :param num: number to name file
    :param instance: name of entity image
    :param backdrop: name of background image
    :param num_instances: number of instances to be pasted onto image
    :return: none
    """
    backdrop = Image.open(backdrop)
    instance = Image.open(instance)
    bd_w, bd_h = backdrop.size
    i_w, i_h = instance.size
    w_split = int(bd_w / num_instances)
    for i in range(num_instances):
        # keep trying to find a position that works
        new_pos_x = random.randint(((i) * w_split), ((i + 1) * w_split) - int(i_w))
        new_pos_y = random.randint(i_h, bd_h - i_h)
        # rotate
        mask = Image.new("L", instance.size, 255)
        # rotate image and mask
        r = random.randint((i * max(0, i - 1)) * w_split, (i * max(0, i - 1)) * w_split * 180)
        instance = instance.rotate(r, expand=True)
        mask = mask.rotate(r, expand=True)
        # paste entity onto backdrop
        backdrop.paste(instance, (new_pos_x, new_pos_y), mask=mask)
    backdrop.save(r"..\..\train_robo\training_scrufe\images\synthetic_robo\images\{}.jpg".format(num), quality=100)


if __name__ == "__main__":
    # generate 200 images, actual project only ended up using 125 due to time
    for i in range(200):
        random_perturb(i)
