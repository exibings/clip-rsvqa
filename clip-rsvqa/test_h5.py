from PIL import Image
import os
import random
import numpy as np
import H5Datasets
import json

dataset_name = "NWPU-Captions"
file_name = "nwpu_captions.h5"
encodings_file_name = "nwpu_captions_encodings.json"


dataset_folder =  os.path.join("datasets", dataset_name)
encodings = json.load(open(os.path.join(dataset_folder, encodings_file_name), "r"))
if dataset_name == "NWPU-Captions":
    class2id = encodings["class2id"]
    id2class = encodings["id2class"]
else:
    id2label = encodings["id2label"]
    label2id = encodings["label2id"]
#print("id2label:", id2label)
#print("label2id:", label2id)
  

print("opening datasets...")
trainDataset = H5Datasets.NwpuCaptionsDataset(os.path.join(dataset_folder, file_name), "train")
valDataset = H5Datasets.NwpuCaptionsDataset(os.path.join(dataset_folder, file_name), "validation")
testDataset = H5Datasets.NwpuCaptionsDataset(os.path.join(dataset_folder, file_name), "test")
idx = random.randint(0, len(trainDataset))
idx = 2
print("number of train samples:", len(trainDataset))
print("number of val samples:", len(valDataset) )
print("number of test samples:", len(testDataset))
#print("number of images:", trainDataset.num_images + valDataset.num_images + testDataset.num_images)
print("Train Sample")
print(trainDataset[idx])
exit()
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