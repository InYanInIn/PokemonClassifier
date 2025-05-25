import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Filter to classify only these 3 types:
SELECTED_TYPES = ['Poison', 'Fire', 'Water']

def load_pokedex(description_file, image_folder, selected_types=None):
    pokedex = pd.read_csv(description_file)
    pokedex.drop('Type2', axis=1, inplace=True, errors='ignore')  # ignore if no Type2
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)
    images = sorted(os.listdir(image_folder))
    images = [os.path.join(image_folder, img) for img in images]
    pokedex['Image'] = images

    if selected_types:
        pokedex = pokedex[pokedex['Type1'].isin(selected_types)].reset_index(drop=True)
    return pokedex

def prepare_dataset(pokedex):
    data_generator = ImageDataGenerator(
        validation_split=0.1,
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator = data_generator.flow_from_dataframe(
        pokedex,
        x_col='Image', y_col='Type1',
        subset='training',
        color_mode='rgba',  # RGBA if images have transparency, else use 'rgb'
        class_mode='categorical',
        target_size=(120, 120),
        shuffle=True,
        batch_size=32
    )

    val_generator = data_generator.flow_from_dataframe(
        pokedex,
        x_col='Image', y_col='Type1',
        subset='validation',
        color_mode='rgba',
        class_mode='categorical',
        target_size=(120, 120),
        shuffle=False
    )

    return train_generator, val_generator

def prepare_network(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(120,120,4)),  # 4 channels for RGBA
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def create_confusion_matrix(model, val_generator):
    val_generator.reset()
    preds = model.predict(val_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes

    labels = list(val_generator.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical', cmap='viridis')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    description_file = 'pokemon.csv'
    image_folder = 'images/images'

    pokedex = load_pokedex(description_file, image_folder, selected_types=SELECTED_TYPES)
    print(pokedex.info())
    print(pokedex.head())

    train_generator, val_generator = prepare_dataset(pokedex)

    model = prepare_network(num_classes=len(SELECTED_TYPES))
    history = model.fit(train_generator, validation_data=val_generator, epochs=30)

    model.save('pokemon_model.keras')
    print("Model saved to pokemon_model.keras")

    plot_history(history)
    print("Training history saved to training_history.png")

    create_confusion_matrix(model, val_generator)
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    main()
