import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

DATASET_PATH = "dataset/"
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"
IMG_SIZE = (96, 96)  # Slightly larger than 64x64

train_datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode=class_mode,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode=class_mode,
    subset="validation"
)

base_model = MobileNetV2(include_top=False, input_shape=(*IMG_SIZE, 3), weights="imagenet")
base_model.trainable = True  # Unfreeze for better learning

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid") if class_mode == "binary"
        else Dense(num_classes, activation="softmax")
])

loss_function = "binary_crossentropy" if class_mode == "binary" else "categorical_crossentropy"
model.compile(optimizer=Adam(1e-4), loss=loss_function, metrics=["accuracy"])

model.fit(
    train_data,
    validation_data=val_data,
    epochs=7,  # Still fast
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)

model.save("image_classifier2.h5")
