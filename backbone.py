# Already trained model available @
# https://github.com/tensorflow/models/tree/master/research/object_detection
# was used as a part of this code.

import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util


def set_model(model_name, label_name):
    model_found = 0

    for file in glob.glob("*"):
        if file == model_name:
            model_found = 1

    # What model to download.
    model_name = model_name
    model_file = model_name + ".tar.gz"
    download_base = "http://download.tensorflow.org/models/object_detection/"

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_ckpt = model_name + "/frozen_inference_graph.pb"

    # List of the strings that is used to add correct label for each box.
    path_to_labels = os.path.join("data", label_name)

    num_classes = 90

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = os.path.join(model_dir, "saved_model")

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    return model, category_index
