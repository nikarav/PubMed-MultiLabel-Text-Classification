import torch
from app.config import settings
from src.util import utils
from functools import partial
import numpy as np


class Model:
    def __init__(self):
        self.tokenizer = Model.setTokenizer()
        self.idx2class = np.array(settings.IDX2CLASS)
        self.threshold = settings.THRESHOLD
        self.model = utils.get_model(
            type=settings.MODEL_NAME,
            n_labels=settings.NUM_LABELS,
            model_path=settings.MODEL_PATH,
            device=torch.device(
                "cpu"
            ),  # Use CPU for the purpose of demonstration. It can easily be adapted for gpu.
        )

        # quantize for faster testing
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()

    @staticmethod
    def setTokenizer():
        tokenizer = utils.getTokenizer(settings.MODEL_NAME)
        return partial(
            tokenizer,
            max_length=settings.MAX_LENGTH,
            truncation=settings.TRUNCATION,
            padding=settings.PADDING,
        )

    def predict(self, text) -> str:
        inputs = self.tokenizer(text)
        with torch.no_grad():
            ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
            masks = torch.tensor(inputs["attention_mask"], dtype=torch.long)

            # batch of 1
            ids = ids.unsqueeze(0)
            masks = masks.unsqueeze(0)
            outputs = self.model(
                ids,
                attention_mask=masks,
            )
        logits = outputs.logits
        logits = logits.squeeze()
        predictions = torch.sigmoid(logits).cpu().numpy()

        predictions = np.where(
            predictions > self.threshold, 1, 0
        )  # transform into binary values
        texted_pred = self.idx2class[
            predictions == 1
        ]  # select the corresponding classes
        return ",".join(texted_pred)
