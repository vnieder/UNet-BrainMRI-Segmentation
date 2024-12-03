import tensorflow as tf

class LossFunctions:
    def __init__(self, smooth=1e-6):
        self.smooth = smooth # Avoids divide by zero error

    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true_flat = tf.keras.backend.flatten(y_true) # Ground truth
        y_pred_flat = tf.keras.backend.flatten(y_pred) # Prediction

        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        denominator = tf.reduce_sum(y_true_flat + y_pred_flat)
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice

    def binary_cross_entropy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.BinaryCrossentropy()
        return bce(y_true, y_pred)

    def iou_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true_flat = tf.keras.backend.flatten(y_true)
        y_pred_flat = tf.keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

    def total_loss(self, y_true, y_pred, bce_weight=0.4, dice_weight=0.4, iou_weight=0.2):
        # Combined loss
        bce = self.binary_cross_entropy(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        iou = self.iou_loss(y_true, y_pred)
        return bce_weight * bce + dice_weight * dice + iou_weight * iou
