import os
import sys
import argparse
import pathlib
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
import matplotlib.pyplot as plt

image_size = 224

def open_image(image_path):
    result = Image.open(image_path)
    result = np.asarray(result)
    return result

def process_image(numpy_image):
    tf_image = tf.convert_to_tensor(numpy_image)
    tf_image = tf.cast(tf_image, tf.float32)
    tf_image= tf.image.resize(tf_image, (image_size, image_size))
    tf_image /= 255
    return tf_image.numpy()

def predict(image, model, top_k = 1):
    image = np.expand_dims(image, axis=0)
    ps = model.predict(image)
    # sort and select top_k elements
    indices = np.flip(np.argpartition(ps, -top_k)[:, -top_k:])
    return ps.take(indices).reshape(top_k), indices.astype(str).reshape(top_k)

def main(image_path, model_path, top_k, class_map_json):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('WARNING')
    tf.device("/GPU:0") if len(tf.config.list_physical_devices("GPU")) > 0 else tf.device("/CPU:0")

    input_image = open_image(parsed_args.image_path)
    input_image = process_image(input_image)
    model = tf.keras.models.load_model(parsed_args.model_path,
                                       custom_objects={'KerasLayer':tfhub.KerasLayer})

    with open(parsed_args.category_names, 'r') as f:
        class_names = json.load(f)
    
    probs, categories = predict(input_image, model, parsed_args.top_k)
    class_labels = [class_names[x] for x in categories]

    return input_image, probs, class_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("image_path", 
                        help="Input image path.",
                        type=pathlib.Path)
    parser.add_argument("model_path", 
                        help="Tensorflow model path.",
                        type=pathlib.Path)
    parser.add_argument("--top_k", 
                        help="Top k predictions.", 
                        nargs="?",
                        default=1,
                        type=int)
    parser.add_argument("--category_names", 
                        help="Overwrite default class names.", 
                        nargs="?",
                        default="class_names_from_dataset.json",
                        type=pathlib.Path)
    parsed_args = parser.parse_args(sys.argv[1:])
    
    input_image, probs, class_labels = main(parsed_args.image_path,
                                            parsed_args.model_path,
                                            parsed_args.top_k,
                                            parsed_args.category_names)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(input_image)
    ax1.axis("off")
    ax2.barh(np.arange(parsed_args.top_k), probs)
    ax2.set_yticks(np.arange(parsed_args.top_k))
    ax2.set_yticklabels(class_labels)    
    plt.tight_layout()
    plt.show(block=True)