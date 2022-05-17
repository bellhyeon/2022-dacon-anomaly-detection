import timm
from models.model.models import Model
from torch import nn


@Model.register("timm_classification")
class TimmClassificationModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        backbone = kwargs["backbone"]
        num_classes = kwargs["num_classes"]

        self.model = timm.create_model(
            backbone, pretrained=True, num_classes=num_classes
        )

    def forward(self, src, tgt=None):
        # pylint: disable=invalid-name
        x = self.model.forward(src)

        return x
