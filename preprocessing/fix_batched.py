import pandas as pd
import os

dataset_folder = os.path.join("datasets", "NWPU-Captions")

train_dataset = pd.read_csv(os.path.join(dataset_folder, "traindf.csv"))
val_dataset = pd.read_csv(os.path.join(dataset_folder, "valdf.csv"))
test_dataset = pd.read_csv(os.path.join(dataset_folder, "testdf.csv"))
merged_df = pd.concat([train_dataset, val_dataset, test_dataset])

train_batched = pd.read_csv(os.path.join(dataset_folder, "batched", "train_batched.csv"))
val_batched = pd.read_csv(os.path.join(dataset_folder, "batched", "val_batched.csv"))
test_batched = pd.read_csv(os.path.join(dataset_folder, "batched", "test_batched.csv"))

def get_image_id(image):
    return merged_df[merged_df["image"] == image]["img_id"].unique().item()

train_batched["img_id"] = train_batched.apply(lambda x: get_image_id(x["image"]), axis="columns")
val_batched["img_id"] = val_batched.apply(lambda x: get_image_id(x["image"]), axis="columns")
test_batched["img_id"] = test_batched.apply(lambda x: get_image_id(x["image"]), axis="columns")

train_batched.to_csv("datasets/NWPU-Captions/batched/train_batched_fixed.csv", index=False)
val_batched.to_csv("datasets/NWPU-Captions/batched/val_batched_fixed.csv", index=False)
test_batched.to_csv("datasets/NWPU-Captions/batched/test_batched_fixed.csv", index=False)