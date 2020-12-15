!#/bin/sh
# steps taken for getting project set up

# utils used to generate tfrecord for this project:
git clone https://github.com/douglasrizzo/detection_util_scripts

# object detection api installation
git clone https://github.com/tensorflow/models.git

# if using windows: install protobuf
# "https://github.com/protocolbuffers/protobuf/releases/protoc-3.14.0-win64.zip"
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
vcpkg install protobuf

# if using linux
# apt-get install protobuf-compiler python-lxml python-pil

cd '../models/research/'
# *** in original training only proto I changed was models/research/object_detection/protos/input_reader.proto
# Line 40, the default shuffle buffer size changed from 2048 to 1000
protoc object_detection/protos/*.proto --python_out=.

# build and install object detection API
python setup.py build
python setup.py install

# test installation
cd '../object_detection/builders/'
python model_builder_tf2_test.py







