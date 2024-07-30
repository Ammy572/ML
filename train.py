import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a simple CNN model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Function to create labels manually
def create_labels(data_path, num_classes):
    # Create a label mapping based on filenames or manually provided labels
    # For this example, we assume the labels are known and created separately
    labels = []  # This should be a list of labels corresponding to your images
    image_files = [f for f in os.listdir(data_path) if f.endswith(('jpg', 'jpeg', 'png'))]
    num_samples = len(image_files)

    for i in range(num_samples):
        label = np.zeros(num_classes)
        # Assign label manually based on image name or external source
        # For simplicity, assume the label for all images is the same for this example
        label[0] = 1  # Example: all images are labeled as class 0
        labels.append(label)
    
    return np.array(labels), image_files

# Data generator function
def data_generator(image_files, labels, data_path, model_image_size, batch_size):
    num_samples = len(image_files)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_files = image_files[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            images = np.array([tf.image.resize(tf.image.decode_image(tf.io.read_file(os.path.join(data_path, f))), model_image_size[:2]) / 255.0 for f in batch_files])
            yield images, batch_labels

# Main training script
def main(args):
    model_image_size = (608, 608, 3)
    num_classes = 5  # Number of classes (bus, car, bicycle, motorcycle, none)

    # Build and compile model
    model = build_model(model_image_size, num_classes)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Create labels
    train_labels, train_files = create_labels(args.train_data_path, num_classes)
    val_labels, val_files = create_labels(args.val_data_path, num_classes)

    # Data generators
    train_generator = data_generator(train_files, train_labels, args.train_data_path, model_image_size, args.batch_size)
    val_generator = data_generator(val_files, val_labels, args.val_data_path, model_image_size, args.batch_size)

    # ModelCheckpoint callback
    checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Train model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_files) // args.batch_size,
        validation_data=val_generator,
        validation_steps=len(val_files) // args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint]
    )

    model.save(args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--model_path", type=str, default="yolo_model.keras", help="Path to save the trained model")
    args = parser.parse_args()
    
    main(args)
