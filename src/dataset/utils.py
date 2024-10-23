import json
from typing import Union
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image


class CocoUtils:
    def __init__(self, root_folder_path):
        self.root_folder_path = root_folder_path

    def convert_to_dataframe(self, file_caption_json_name: str):
        annotation = self.root_folder_path + f"/annotations/{file_caption_json_name}"
        with open(annotation, "r") as f:
            data = json.load(f)
            data = data["annotations"]
        samples = []
        for item in data:
            image_path = "%012d.jpg" % item["image_id"]
            samples.append([image_path, item["caption"]])
        df = pd.DataFrame(samples, columns=["image_path", "caption"])
        df["image_path"] = df["image_path"].apply(
            lambda x: self.root_folder_path + f"/train2017/{x}"
        )
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def plot_n_images_with_captions(df: pd.DataFrame, n):
        fig, axs = plt.subplots(1, n, figsize=(20, 5))
        for i in range(n):
            image_path = df.iloc[i]["image_path"]
            caption = df.iloc[i]["caption"]
            img = Image.open(image_path)
            axs[i].imshow(img)
            axs[i].set_title(caption)
            axs[i].axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_train_val_df(
        df: pd.DataFrame, val_size: float
    ) -> Union[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = train_test_split(df, test_size=val_size, random_state=42)
        train_df, val_df = (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
        )
        return train_df, val_df
