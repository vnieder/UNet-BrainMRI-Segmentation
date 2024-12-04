from src.model.unet import UNet
from src.dataloader.dataloader import DataLoader
from src.criterion.criterion import LossFunctions
from src.training.training import Trainer
from src.viz.viz import DataVisualization

###
### HYPERPARAMETERS
###
INPUT_SIZE = 256
BASE_FILTERS = 64
LEARNING_RATE = 1e-4
EPOCHS = 50
BATCH_SIZE = 16
DROPOUT_RATE = 0.3
DECAY_RATE = 0.9
DECAY_STEPS = 1000

# Data Paths
train_path = 'brain_tumor_dataset/train'
test_path = 'brain_tumor_dataset/test'
valid_path = 'brain_tumor_dataset/valid'

train_ann_path = 'brain_tumor_dataset/train/_annotations.coco.json'
test_ann_path = 'brain_tumor_dataset/test/_annotations.coco.json'
valid_ann_path = 'brain_tumor_dataset/valid/_annotations.coco.json'

# Save paths
MODEL_PATH = "unet_model_best.weights.h5"
LOGS_PATH = "training_logs.csv"

###
### DATA LOADING
###
print("Loading data . . .")
data_loader = DataLoader(input_size=INPUT_SIZE)

# Parse datasets
train_images, train_masks = data_loader.parse_annotations(train_ann_path, train_path)
valid_images, valid_masks = data_loader.parse_annotations(valid_ann_path, valid_path)
test_images, test_masks = data_loader.parse_annotations(test_ann_path, test_path)

# Create TensorFlow datasets
train_dataset = data_loader.create_dataset(train_images, train_masks, batch_size=BATCH_SIZE, shuffle=True)
valid_dataset = data_loader.create_dataset(valid_images, valid_masks, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = data_loader.create_dataset(test_images, test_masks, batch_size=BATCH_SIZE, shuffle=False)

###
### MODEL BUILDING
###
print("Building U-Net model . . .")
unet = UNet(input_size=INPUT_SIZE, base_filters=BASE_FILTERS, dropout_rate=DROPOUT_RATE)
model = unet.build_model()

# Define loss function
loss_functions = LossFunctions()
total_loss = loss_functions.total_loss

# Instantiate trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    loss=total_loss,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    initial_lr=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    model_save_path=MODEL_PATH,
    logs_path=LOGS_PATH
)

###
### TRAINING
###
print("Starting training . . .")
history = trainer.train()

###
### EVALUATION
###
print("Evaluating model . . .")
model.load_weights(MODEL_PATH)
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

###
### VISUALIZATION
###
print("Visualizing Data . . .")
visualizer = DataVisualization(model=model)

visualizer.plot_train_val_loss(LOGS_PATH)
visualizer.plot_train_val_accuracy(LOGS_PATH)

visualizer.visualize_predictions(test_dataset, num_examples=3)
