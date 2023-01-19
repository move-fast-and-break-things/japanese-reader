from pathlib import Path
import torch
import flash
from flash.image import ImageClassificationData, ImageClassifier


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


model = ImageClassifier(backbone="resnet34", labels=datamodule.labels)


trainer = flash.Trainer(max_epochs=7, gpus=torch.cuda.device_count(), auto_lr_find=True)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")


datamodule = ImageClassificationData.from_files(
    predict_files=[
        "data/katakana/train/ba/ba_Gyate-Luminescence_dakutrue.jpeg",
    ],
    batch_size=1,
)
predictions = trainer.predict(model, datamodule=datamodule, output="labels")
print(predictions)


trainer.save_checkpoint("resnet34_8epochs.pt")
