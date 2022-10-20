import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testPhiliDataset = pd.read_csv(os.path.join("datasets", "RSVQA-HR", "testdf_phili.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})

trainDataset["label"] = trainDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
trainDataset["label"] = trainDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(
    lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
validationDataset["label"] = validationDataset.apply(
    lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testDataset["label"] = testDataset.apply(lambda x: utils.count_func(x["label"], x["category"]), axis="columns")
testPhiliDataset["label"] = testPhiliDataset.apply(lambda x: utils.area_func(x["label"], x["category"]), axis="columns")
testPhiliDataset["label"] = testPhiliDataset.apply(
    lambda x: utils.count_func(x["label"], x["category"]), axis="columns")

label2id, id2label = utils.encodeDatasetLabels("RSVQA-HR", trainDataset, validationDataset, testDataset, testPhiliDataset)
encodings = {"label2id": label2id, "id2label": id2label}
with open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_encodings.json"), "w") as encodings_file:
    json.dump(encodings, encodings_file)
with open(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr_id2label.json"), "w") as id2label_file:
    json.dump(id2label, id2label_file)
trainDataset.replace(label2id, inplace=True)
validationDataset.replace(label2id, inplace=True)
testDataset.replace(label2id, inplace=True)
testPhiliDataset.replace(label2id, inplace=True)

image_count = utils.jpgOnly("RSVQA-HR")
if utils.verifyImages("RSVQA-HR"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQA-HR", "rsvqa_hr.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQA-HR", hfile, "train", trainDataset)
        del trainDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "validation", validationDataset)
        del validationDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "test", testDataset)
        del testDataset
        utils.createDatasetSplit("RSVQA-HR", hfile, "test_phili", testPhiliDataset)
        del testPhiliDataset
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32")
        images = os.listdir(os.path.join("datasets", "RSVQA-HR", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQA-HR", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
else:
    print("Missing images.")