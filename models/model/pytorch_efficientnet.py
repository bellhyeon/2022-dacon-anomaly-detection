from efficientnet_pytorch import EfficientNet
from models.model.models import Model
from torch import nn


@Model.register("pytorch_efficientnet")
class PytorchEfficientNet(Model):
    def __init__(self, **kwargs):
        super().__init__()

        backbone = kwargs["backbone"]
        num_classes = kwargs["num_classes"]
        dropout_rate = kwargs["dropout_rate"]
        self.model = EfficientNet.from_pretrained(
            model_name=backbone, num_classes=num_classes, dropout_rate=dropout_rate
        )

    def forward(self, src, tgt=None):
        x = self.model(src)
        return x
