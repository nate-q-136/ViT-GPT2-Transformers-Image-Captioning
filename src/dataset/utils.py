import json
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os


class BaseDataUtils:
    def __init__(self, root_folder_path):
        self.root_folder_path = root_folder_path

    def convert_to_dataframe(self, file_caption_json_name: str) -> pd.DataFrame:
        pass

    @staticmethod
    def plot_n_images_with_captions(df: pd.DataFrame, n):
        fig, axs = plt.subplots(1, n, figsize=(20, 5))
        for i in range(n):
            index_random = np.random.randint(0, len(df))
            image_path = df.iloc[index_random]["image_path"]
            caption = df.iloc[index_random]["caption"]
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


class CocoUtils(BaseDataUtils):
    def convert_to_dataframe(
        self, folder_images, file_caption_json_name: str
    ) -> pd.DataFrame:
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
            lambda x: self.root_folder_path + f"/{folder_images}/{x}"
        )
        df = df.reset_index(drop=True)
        return df


class Flickr8kUtils(BaseDataUtils):
    def convert_to_dataframe(
        self, file_caption_txt: str, folder_images: str
    ) -> pd.DataFrame:
        """
        Read caption txt file
        """
        df = pd.read_csv(
            os.path.join(self.root_folder_path, file_caption_txt),
            header=None,
            names=["image", "caption"],
        )
        # delete first row
        df = df.iloc[1:]
        df["image_path"] = self.root_folder_path + f"/{folder_images}/" + df["image"]
        df.drop(columns=["image"], inplace=True)
        df = df.reset_index(drop=True)
        return df


class Flickr30kUtils(BaseDataUtils):
    def convert_to_dataframe(
        self, file_caption_txt: str, folder_images: str
    ) -> pd.DataFrame:
        """
        Read caption txt file
        """
        df = pd.read_csv(
            os.path.join(self.root_folder_path, file_caption_txt),
            header=None,
            names=["image", "comment_number", "caption"],
        )
        # delete first row
        df = df.iloc[1:]
        df["image_path"] = self.root_folder_path + f"/{folder_images}/" + df["image"]
        df.drop(columns=["image"], inplace=True)
        df = df.reset_index(drop=True)
        return df


# if __name__ == "__main__":
#     flickr = Flickr8kUtils(root_folder_path="/Volumes/Untitled 2 1/3-CNN-Tensorflow/26-Pytorch/10-image-captioning/flickr8k")
#     df = flickr.convert_to_dataframe(file_caption_txt="captions.txt")
#     print(df.head())
#     flickr.plot_n_images_with_captions(df, 5)
