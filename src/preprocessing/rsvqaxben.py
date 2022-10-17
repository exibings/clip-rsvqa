import os
import utils
import h5py
from PIL import Image
import pandas as pd
import json

trainDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "traindf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
validationDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "valdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
testDataset = pd.read_csv(os.path.join("datasets", "RSVQAxBEN", "testdf.csv"), sep=",").drop(columns="mode").rename(columns={"answer": "label"})
label2id, id2label = utils.encodeDatasetLabels("RSVQAxBEN", trainDataset, validationDataset, testDataset)
with open(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben_label2id.json"), "w") as label2id_file:
    json.dump(label2id, label2id_file)
with open(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben_id2label.json"), "w") as id2label_file:
    json.dump(id2label, id2label_file)
trainDataset.replace(label2id, inplace=True)
validationDataset.replace(label2id, inplace=True)
testDataset.replace(label2id, inplace=True)

image_count = utils.jpgOnly("RSVQAxBEN")
if utils.verifyImages("RSVQAxBEN"):
    # create .h5 file
    with h5py.File(os.path.join("datasets", "RSVQAxBEN", "rsvqaxben.h5"), "w") as hfile:
        utils.createDatasetSplit("RSVQAxBEN", hfile, "train", trainDataset)
        del trainDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "validation", validationDataset)
        del validationDataset
        utils.createDatasetSplit("RSVQAxBEN", hfile, "test", testDataset)
        del testDataset
        print("Processing images and adding them to the dataset...")
        pixel_values = hfile.create_dataset("pixel_values", (image_count, 5, 3, 224, 224), "float32", compression="gzip", compression_opts=2)
        images = os.listdir(os.path.join("datasets", "RSVQAxBEN", "images"))
        for image_idx in range(len(images)):
            image = utils.feature_extractor(utils.patchImage(os.path.join("datasets", "RSVQAxBEN", "images",
                                                                    images[image_idx])), return_tensors="np", resample=Image.Resampling.BILINEAR)
            pixel_values[int(images[image_idx].split(".")[0])] = image.pixel_values
else:
    print("Missing images.")