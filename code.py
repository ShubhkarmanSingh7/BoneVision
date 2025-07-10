import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import os, shutil
from sklearn.model_selection import train_test_split

# Update this to your dataset root
original_data_dir = r"E:\Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures\raw"
output_base_dir = r"E:\Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures\dataset"

# Desired structure
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.15, 0.15]

# Make folders
classes = os.listdir(original_data_dir)
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_base_dir, split, cls), exist_ok=True)

# Split and copy files
for cls in classes:
    files = os.listdir(os.path.join(original_data_dir, cls))
    train_files, testval_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(testval_files, test_size=0.5, random_state=42)

    for split, file_list in zip(splits, [train_files, val_files, test_files]):
        for file in file_list:
            src = os.path.join(original_data_dir, cls, file)
            dst = os.path.join(output_base_dir, split, cls, file)
            shutil.copy2(src, dst)

import os
data_path = r"E:\Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures\dataset"
print("Checking path:", path)
print("Path exists:", os.path.exists(path))

img_size = (224, 224)
batch_size = 32
#data_path = "E:\Bone Fracture X-ray Dataset Simple vs. Comminuted Fractures\dataset"  # UPDATE with your folder path

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    os.path.join(data_path, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    os.path.join(data_path, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = datagen.flow_from_directory(
    os.path.join(data_path, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

class_labels = list(train_gen.class_indices.keys())
print("Class labels:", class_labels)

def build_custom_cnn():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_alexnet():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(96, (11, 11), strides=4, activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_transfer_model(base_model_func):
    base_model = base_model_func(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=output)
def train_model(model, name):
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print(f"\nTraining {name}...")
    history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)
    return model, history

def evaluate_model(model, name):
    test_gen.reset()
    preds = model.predict(test_gen)
    y_pred = (preds > 0.5).astype(int)
    y_true = test_gen.classes

    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

models = {
    'CustomCNN': build_custom_cnn(),
    'AlexNet': build_alexnet(),
    'DenseNet121': build_transfer_model(DenseNet121),
    'EfficientNetB0': build_transfer_model(EfficientNetB0),
    'ResNet50': build_transfer_model(ResNet50),
}

histories = {}
trained_models = {}

for name, model in models.items():
    trained_model, history = train_model(model, name)
    trained_models[name] = trained_model
    histories[name] = history
    evaluate_model(trained_model, name)

def plot_histories(histories):
    for name, history in histories.items():
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{name} Training Metrics", fontsize=14)

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

  def compare_models(histories):
    plt.figure(figsize=(14, 5))

    # Validation Accuracy Comparison
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history.history['val_accuracy'], label=name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Validation Loss Comparison
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history.history['val_loss'], label=name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
print_final_metrics(histories)
def print_final_metrics(histories):
    print("\nFinal Validation Accuracy and Loss:")
    print("-" * 40)
    for name, history in histories.items():
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        print(f"{name:<15} | Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f}")
print_final_metrics(histories)
