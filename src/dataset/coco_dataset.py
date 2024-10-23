from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, ViTFeatureExtractor
from PIL import Image
import albumentations as A
import numpy as np


class ViTGPT2Dataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer: AutoTokenizer,
        feature_extractor: ViTFeatureExtractor,
        max_length: int = 128,
        transform: A = None,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df["caption"][idx]
        image_path = self.df["image_path"][idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = np.array(image)
            augs = self.transform(image=image)
            image = augs["image"]
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        tokenized_caption = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.max_length
        ).input_ids
        tokenized_caption = [
            token if token != self.tokenizer.pad_token_id else -100
            for token in tokenized_caption
        ]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(tokenized_caption),
        }
        return encoding
