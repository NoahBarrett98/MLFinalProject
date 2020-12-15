#!/bin/sh
# Scipt to show how dataset was built:

# install annotation software, this one seems most common and is very handy
pip install labelIMG

# create dataset
mkdir ./data/training/images
# build images from a backdrop and robot image
# by changing up these two images, any arbitrary obj/background pair could be generated
python ./src/preprocess/build_imgs.py

# annotate dataset
labelIMG ./data/training/images

# seperate files into images and xml
# python ../src/preprocess/split_xml_img -dir=./images

python ./src/api_scripts/partition_dataset.py -x -i ./data/training/images/images/ -r 0.1

# generate csv for data
mkdir ./data/training/annotations
python generate_csv.py xml ./data/training/images/xml ./data/training/annotations/train.csv

# generate labels for dataset
python generate_pbtxt.py csv ./data/training/annotations/train.csv ./data/training/annotations/label_map.pbtxt


# train data
python ./src/api_scripts/generate_tfrecord.py -x data/training/images/images/train -l data/training/annotations/label_map.pbtxt -o data/training/annotations/train.tfrecord
# test data
python ./src/api_scripts/generate_tfrecord.py -x data/training/images/images/test -l data/training/annotations/label_map.pbtxt -o data/training/annotations/test.tfrecord