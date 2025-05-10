import os
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = 'dataset/'
IMG_SIZE = (96, 96)  # Match training size
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        return

    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"Error: Corrupted image - {image_path}")
        return

    model = tf.keras.models.load_model("image_classifier2.h5")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image - {image_path}")
        return

    img = cv2.resize(img, IMG_SIZE)  # Match training size
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = sorted(os.listdir(DATASET_PATH))  # Alphabetical order = consistent

    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]

    print(f"The model has determined: {predicted_class}")
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"The model has determined: {predicted_class}")
    plt.axis('off')
    plt.show()

predict_image("snake.png")
