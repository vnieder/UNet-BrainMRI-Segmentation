import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model

class UNet:
    def __init__(self, input_size=256, base_filters=64, dropout_rate=0.3):
        # Default input, filters, and dropout are 256, 64, and .3 respectively
        self.input_size = input_size
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate

    def build_model(self):
        inputs = Input(shape=(self.input_size, self.input_size, 3))  # Default input shape: (256, 256, 3)

        # Encoder - Downscaling
        c1 = Conv2D(self.base_filters, (3, 3), activation='relu', padding='same')(inputs)
        c1 = BatchNormalization()(c1) # Batch Normalization to allieviate overfitting
        c1 = Conv2D(self.base_filters, (3, 3), activation='relu', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(self.base_filters * 2, (3, 3), activation='relu', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(self.base_filters * 2, (3, 3), activation='relu', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(self.base_filters * 4, (3, 3), activation='relu', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(self.base_filters * 4, (3, 3), activation='relu', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(self.dropout_rate)(c3) # Dropout to allieviate overfitting
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(self.base_filters * 8, (3, 3), activation='relu', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Conv2D(self.base_filters * 8, (3, 3), activation='relu', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(self.dropout_rate)(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = Conv2D(self.base_filters * 16, (3, 3), activation='relu', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Conv2D(self.base_filters * 16, (3, 3), activation='relu', padding='same')(c5)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(self.dropout_rate)(c5)

        # Decoder - Upscaling
        u6 = Conv2DTranspose(self.base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = Concatenate()([u6, c4]) # Concatenate for skip connection
        c6 = Conv2D(self.base_filters * 8, (3, 3), activation='relu', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Conv2D(self.base_filters * 8, (3, 3), activation='relu', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(self.base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(self.base_filters * 4, (3, 3), activation='relu', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Conv2D(self.base_filters * 4, (3, 3), activation='relu', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(self.base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(self.base_filters * 2, (3, 3), activation='relu', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Conv2D(self.base_filters * 2, (3, 3), activation='relu', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(self.base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(self.base_filters, (3, 3), activation='relu', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Conv2D(self.base_filters, (3, 3), activation='relu', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        # Output
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)  # Sigmoid activation for prediction

        model = Model(inputs, outputs, name="UNet")
        return model