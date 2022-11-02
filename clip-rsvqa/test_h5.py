from PIL import Image
import os
import random
import numpy as np
from H5Dataset import H5Dataset
import json

id2label = json.load(open(os.path.join("datasets","RSVQA-LR", "rsvqa_lr_id2label.json"), "r"))
label2id = json.load(open(os.path.join("datasets","RSVQA-LR", "rsvqa_lr_label2id.json"), "r"))

print("id2label:", id2label)
print("label2id:", label2id)
  

print("opening datasets...")
trainDataset = H5Dataset(os.path.join("datasets", "RSVQA-LR", "rsvqa_lr.h5"), "train", "baseline")
print("datasets opened!")


idx = random.randint(0, 772)
print("line:", idx+2)
print("number of samples:", len(trainDataset))
print("Train")
print("\tcategory:", trainDataset["category"][idx].decode("utf-8"))
print("\timg id:", trainDataset["img_id"][idx])
print("\tquestion:", trainDataset["question"][idx].decode("utf-8"))
print("\tinput ids:", trainDataset["input_ids"][idx])
print("\tattention mask:", trainDataset["attention_mask"][idx])
print("\tlabel:", id2label[str(trainDataset["label"][idx])]) 
print("Grabbing image", trainDataset["img_id"][idx])
pixel_values = trainDataset["pixel_values"]*255
pixel_values = pixel_values.astype(np.uint8)
patch1 = pixel_values[0]
patch1 = np.moveaxis(patch1, 0, -1)
patch2 = pixel_values[1]
patch2 = np.moveaxis(patch2, 0, -1)
patch3 = pixel_values[2]
patch3 = np.moveaxis(patch3, 0, -1)
patch4 = pixel_values[3]
patch4 = np.moveaxis(patch4, 0, -1)
full_image = pixel_values[4]
full_image = np.moveaxis(full_image, 0, -1)
Image.fromarray(patch1).save(os.path.join("datasets", "RSVQA-LR", "patch1.jpg"))
Image.fromarray(patch2).save(os.path.join("datasets", "RSVQA-LR", "patch2.jpg"))
Image.fromarray(patch3).save(os.path.join("datasets", "RSVQA-LR", "patch3.jpg"))
Image.fromarray(patch4).save(os.path.join("datasets", "RSVQA-LR", "patch4.jpg"))
Image.fromarray(full_image).save(os.path.join("datasets", "RSVQA-LR", "full_image.jpg"))