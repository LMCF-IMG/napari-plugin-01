#%%
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import *
from PIL import Image
import timm
import matplotlib.pyplot as plt

#%% For release...
def Run_Transfer_Learning_Train(directory):
    path_to_images = "d://Programovani//MachineLearning2024_Prague_Conference_Transfer_Learning//images"
    path_to_csv = os.path.split(path_to_images)[0]  # dir one up
    print(path_to_csv)

    train_df = pd.read_csv(os.path.join(path_to_csv, "train.csv"))
    test_df = pd.read_csv(os.path.join(path_to_csv, "test.csv"))

    lbl_map = {lbl:i for i, lbl in enumerate(sorted(set(train_df["label"])))}
    inv_lbl_map = {i:lbl for lbl, i in lbl_map.items()}
    print(lbl_map)

    train_df["image"] = path_to_images + train_df["image_id"]
    train_df["fold"] = train_df.index % 5
    test_df["image"] = path_to_images + test_df["image_id"]

    print(train_df.head())

    pass

Run_Transfer_Learning_Train("d://Programovani//MachineLearning2024_Prague_Conference_Transfer_Learning//images")

#%% For Debug
path_to_images = "d://Programovani//MachineLearning2024_Prague_Conference_Transfer_Learning//images"
path_to_csv = os.path.split(path_to_images)[0]  # dir one up
print(path_to_csv)

train_df = pd.read_csv(os.path.join(path_to_csv, "train.csv"))
test_df = pd.read_csv(os.path.join(path_to_csv, "test.csv"))

lbl_map = {lbl:i for i, lbl in enumerate(sorted(set(train_df["label"])))}
inv_lbl_map = {i:lbl for lbl, i in lbl_map.items()}
print(lbl_map)

# !!! tady mi to blbne, ve VS Code je to tak ok, v napari se to nějak špatně dává dohromady
train_df["image"] = path_to_images + train_df["image_id"]
train_df["fold"] = train_df.index % 5
test_df["image"] = path_to_images + test_df["image_id"]

print(train_df.head())

# %%
