import gradio as gr
from flash.image import ImageClassifier
import lightning as L
from lightning.app.components.serve import ServeGradio
from flash.image import ImageClassificationData
from flash import Trainer


class LitGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", source="canvas", shape=(196, 196), label="Draw a glyph")
    outputs = gr.outputs.Label(label="Prediction")

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, image) -> str:
        trainer = Trainer()
        datamodule = ImageClassificationData.from_images(
            predict_images=[image],
            transform_kwargs={"image_size": (196, 196)},
            batch_size=1,
            )
        predictions = trainer.predict(self.model, datamodule=datamodule, output="labels")
        return predictions[0][0]
    
    def build_model(self):
        model = ImageClassifier.load_from_checkpoint("resnet34_8epochs.pt")
        self.ready = True
        return model

app = L.LightningApp(LitGradio())