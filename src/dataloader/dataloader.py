import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

class DataLoader:
    def __init__(self, input_size):
        self.input_size = input_size

    def parse_annotations(self, annotation_path, images_path):
        coco = COCO(annotation_path)
        image_ids = coco.getImgIds()
        image_files = []
        mask_files = []
        
        # Loads all images and respective annotations
        for img_id in image_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_path, img_info['file_name'])

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Generates Mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for ann in anns:
                mask = np.maximum(mask, coco.annToMask(ann))

            image_files.append(img_path)
            mask_files.append(mask)
        return image_files, mask_files

    def preprocess_image(self, image_path, mask):
        # Preprocesses image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (self.input_size, self.input_size)) # Shape of image is (height, width, 3)
        image = tf.cast(image, tf.float32) / 255.0
        
        # Preprocesses mask
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, (self.input_size, self.input_size)) # Shape of mask is (height, width, 1)
        mask = tf.cast(mask, tf.float32)
        return image, mask

    def create_dataset(self, image_paths, masks, batch_size=16, shuffle=True):
        def generator():
            for img_path, mask in zip(image_paths, masks):
                yield img_path, mask

        # Define Tensorflow Dataset Object
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None, None), dtype=tf.uint8)
            )
        )
        
        dataset = dataset.map(
            lambda img_path, mask: self.preprocess_image(img_path, mask),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Randomly shuffles the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=100)
    
        # Batches and dynamically prefetches data
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
