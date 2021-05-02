import numpy as np
import json

import tensorflow as tf

def get_class_names(class_map_file):
    with open(class_map_file, 'r') as f:
        class_names = json.load(f)
    return class_names
    
def process_image(image_array):
    image_size = 224
    test_image = tf.cast(image_array, tf.float32)
    test_image = tf.image.resize(test_image, (image_size, image_size))
    test_image /= 255
    return test_image