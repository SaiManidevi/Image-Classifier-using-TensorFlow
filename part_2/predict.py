import argparse
import numpy as np
import json

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from utils import get_class_names, process_image 

def main():
    parser = argparse.ArgumentParser(description='Flower Image Classifier')
    parser.add_argument('image_path', action='store', type=str, help='Enter the image path:')
    parser.add_argument('saved_model', action='store', type=str, help='Enter the name of the model:')
    parser.add_argument('top_k', action='store', type=str, help='Enter k:')
    args = parser.parse_args()
    print(args)
    input_image_path = args.image_path
    model = args.saved_model
    top_k = args.top_k
    loaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    probs, labels = predict(input_image_path, loaded_model, top_k)
    class_names = get_class_names()
    print('Top', top_k, 'predictions for given image')
    for i in range(int(top_k)):
        print('Class name: ',class_names.get(str(labels[i])).capitalize(),
             ' Prediction probability:', probs[i])
        
    
def predict(image_path, model, top_k):
    # Convert given image as an np array
    image_as_array = np.asarray(Image.open(image_path))
    # pre-process the image
    preprocessed_image = process_image(image_as_array)
    # add extra dimension to match the model input size
    input_image = np.expand_dims(preprocessed_image, axis= 0)
    # Get the Predictions for the processed image
    predictions = model.predict(input_image)
    # Get the Probabilities and Labels
    prob, labels = tf.math.top_k(predictions, int(top_k))
    labels += 1
    return prob.numpy()[0], labels.numpy()[0]

if __name__ == "__main__":
    main()

