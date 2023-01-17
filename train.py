from pathlib import Path
import torch
import flash
from flash.image import ImageClassificationData, ImageClassifier
import del_symbol
# 1. Create the DataModule

data_dir = "data/katakana"

images_path = Path(data_dir)
train_path = str(images_path/"train")
validation_path = str(images_path/"validation")

datamodule = ImageClassificationData.from_folders(
    train_folder=train_path,
    val_folder=validation_path,
    batch_size=3,
    transform_kwargs={"image_size": (196, 196), "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", labels=datamodule.labels)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! ants or bees?
datamodule = ImageClassificationData.from_files(
    predict_files=[
        "data/katakana/train/ba/ba_Gyate-Luminescence_dakutrue.jpeg",
    ],
    batch_size=1,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")
