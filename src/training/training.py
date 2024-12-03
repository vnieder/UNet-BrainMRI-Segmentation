import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class Trainer:
    def __init__(self, model, train_dataset, valid_dataset, loss, epochs, batch_size, initial_lr, decay_steps, decay_rate, model_save_path="unet_model_best.weights.h5", logs_path="training_logs.csv"):
        # Default save path and log path are unet_model_best.weights.h5 and training_logs.csv respectively
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.loss = loss
        self.model_save_path = model_save_path
        self.logs_path = logs_path
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        # Exponential Decay learning rate scheduler
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True
        )

        # Model includes custom loss function and learning rate scheduler
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=self.loss,
            metrics=["accuracy"]
        )

        # Callbacks include ModelCheckpoint, CSVLogger, and EarlyStopping
        callbacks_list = [
            ModelCheckpoint(self.model_save_path, monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min"),
            CSVLogger(self.logs_path),
            EarlyStopping(monitor="val_loss", patience=5, verbose=1)
        ]

        # Training loop
        return self.model.fit(
            self.train_dataset,
            validation_data=self.valid_dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )