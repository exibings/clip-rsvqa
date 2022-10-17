import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQA-LR", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})

trainDataset["label"] = trainDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
trainDataset["label"] = trainDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")

label2id, id2label = utils.encodeDatasetLabels("RSVQA-LR", trainDataset, validationDataset, testDataset)
with open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_label2id.json"), "w") as label2id_file:
    json.dump(label2id, label2id_file)
with open(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr_id2label.json"), "w") as id2label_file:
    json.dump(id2label, id2label_file)
trainDataset.replace(label2id, inplace=True)
validationDataset.replace(label2id, inplace=True)
testDataset.replace(label2id, inplace=True)

image_count = utils.jpgOnly("RSVQA-LR")
if utils.verifyImages("RSVQA-LR"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQA-LR", hfile, "train", trainDataset)
        del trainDataset
        utils.createDatasetSplit("RSVQA-LR", hfile, "validation", validationDataset)
        del validationDataset
        utils.createDatasetSplit("RSVQA-LR", hfile, "test", testDataset)
        del testDataset
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "RSVQA-LR", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-LR", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
else:
    print("Missing images.")