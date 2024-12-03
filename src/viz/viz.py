import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

class DataVisualization:
    def __init__(self, model=None):
        self.model = model

    def set_model(self, model):
        self.model = model

    def visualize_predictions(self, dataset, num_examples=2):
        for i, (images, masks) in enumerate(dataset.take(num_examples)):
            preds = self.model.predict(images)
            preds_binary = tf.cast(preds > 0.5, tf.float32)

            for j in range(images.shape[0]):
                plt.figure(figsize=(12, 4))

                # Input
                plt.subplot(1, 3, 1)
                plt.title("Input Image")
                plt.imshow(tf.keras.utils.array_to_img(images[j]))

                # Ground truth mask
                plt.subplot(1, 3, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(tf.keras.utils.array_to_img(masks[j]), cmap="gray")

                # Predicted mask
                plt.subplot(1, 3, 3)
                plt.title("Predicted Mask")
                plt.imshow(tf.keras.utils.array_to_img(preds_binary[j]), cmap="gray")

                plt.tight_layout()
                plt.show()

    def plot_train_val_loss(self, log_path):
        history = pd.read_csv(log_path)
        
        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["loss"], label="Train Loss")
        plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_train_val_accuracy(self, log_path):
        history = pd.read_csv(log_path)

        plt.figure(figsize=(10, 5))
        plt.plot(history["epoch"], history["accuracy"], label="Train Accuracy")
        plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
