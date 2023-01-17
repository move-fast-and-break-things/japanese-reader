from functools import partial
import gradio as gr
import requests
import torch
from PIL import Image
from flash.image import ImageClassifier

import lightning as L
from lightning.app.components.serve import ServeGradio
from flash.image import ImageClassificationData
from flash import Trainer


class LitGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", source="canvas", label="Draw a glyph")
    outputs = gr.outputs.Label(label="Prediction")
    #demo_img = "https://imageio.forbes.com/specials-images/imageserve/60abf319b47a409ca17f4a3f/Pedestrians-cross-Broadway--in-the-SoHo-neighborhood-in-New-York--United-States--May/960x0.jpg?format=jpg&width=960"
    #img = Image.open(requests.get(demo_img, stream=True).raw)
    #img.save("960x0.jpg")
    #examples = [["960x0.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, image) -> str:
        trainer = Trainer()
        datamodule = ImageClassificationData.from_images(
        predict_files=[image]
        )
        predictions = trainer.predict(self.model, datamodule=datamodule)
        return predictions[0]
    
    def build_model(self):
        model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")
        self.ready = True
        return model

app = L.LightningApp(LitGradio())