import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models

train_dir = "food101/images"
TARGET_CLASSES = ["apple_pie", "churros"]
REMOVE_RATIO = 0.05

def reduce_dataset(directory, target_classes, remove_ratio):
    for class_name in target_classes:
        class_path = os.path.join(directory, class_name)
        if not os.path.exists(class_path):
            continue
        images = os.listdir(class_path)
        num_to_remove = int(len(images) * remove_ratio)
        images_to_remove = random.sample(images, num_to_remove)
        for img in images_to_remove:
            img_path = os.path.join(class_path, img)
            if os.path.isfile(img_path):
                os.remove(img_path)

reduce_dataset(train_dir, TARGET_CLASSES, REMOVE_RATIO)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2)
])

def preprocess_data(dataset):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(data_augmentation(x)), y)).prefetch(buffer_size=AUTOTUNE)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=123
)

train_dataset = preprocess_data(train_dataset)
validation_dataset = preprocess_data(validation_dataset)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr]
)

base_model.trainable = True
for layer in base_model.layers[:-30]: 
    layer.trainable = False

fine_tune_lr = 1e-5
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),loss="categorical_crossentropy", metrics=["accuracy"])

history_finetune = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr]
)

model.save("food101_model.h5")