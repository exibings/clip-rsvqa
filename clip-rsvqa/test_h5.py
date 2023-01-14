from PIL import Image
import os
import random
import numpy as np
import H5Datasets
import json
dataset_name = "NWPU-Captions"
file_name = "nwpu_captions.h5"
encodings_file_name = "nwpu_captions_metadata.json"


dataset_folder =  os.path.join("datasets", dataset_name)
encodings = json.load(open(os.path.join(dataset_folder, encodings_file_name), "r"))
if dataset_name == "NWPU-Captions":
    id2class = encodings["id2class"]
    class2id = encodings["class2id"]
else:
    id2label = encodings["id2label"]
    label2id = encodings["label2id"]
#print("id2label:", id2label)
#print("label2id:", label2id)  

print("opening datasets...")
dataset = H5Datasets.NwpuCaptionsDataset(file_name, "train", augment_images=False)
idx = random.randint(0, len(dataset))
idx = 0
print("number of samples:", len(dataset))
#print("number of images:", trainDataset.num_images + valDataset.num_images + testDataset.num_images)
#print("num_labels:", trainDataset.num_labels)
print("Sample:")
print(dataset[idx])
print("translated label:", id2class[str(dataset[idx]["class"])])
print("Grabbing image", dataset[idx]["img_id"])
pixel_values = dataset[idx]["pixel_values"]*255
pixel_values = pixel_values.astype(np.uint8)
"""patch1 = pixel_values[0]
patch1 = np.moveaxis(patch1, 0, -1)
patch2 = pixel_values[1]
patch2 = np.moveaxis(patch2, 0, -1)
patch3 = pixel_values[2]
patch3 = np.moveaxis(patch3, 0, -1)
patch4 = pixel_values[3]
patch4 = np.moveaxis(patch4, 0, -1)
full_image = pixel_values[4]
print(pixel_values)
full_image = np.moveaxis(pixel_values, 0, -1)
Image.fromarray(full_image).save(os.path.join("datasets", "NWPU-Captions", "full_image.jpg"))"""