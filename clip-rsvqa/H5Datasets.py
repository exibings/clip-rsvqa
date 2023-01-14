import torch
import h5py
import numpy as np
from PIL import Image
import os
import utils
import json

class RsvqaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, model_type):
        self.folder = os.path.join("datasets", dataset_name)
        if dataset_name == "RSVQA-LR":
            self.h5_file_name = "rsvqa_lr.h5"
            self.metadata_file_name = "rsvqa_lr_metadata.json"
        elif dataset_name == "RSVQA-HR":
            self.h5_file_name = "rsvqa_hr.h5"
            self.metadata_file_name = "rsvqa_hr_metadata.json"
        self.h5_file_path = os.path.join(self.folder, self.h5_file_name)
        self.metadata_file_path = os.path.join(self.folder, self.metadata_file_name)
        self.dataset = None
        self.split = split
        self.model_type = model_type
        with h5py.File(os.path.join(self.folder, self.h5_file_name), 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["category"]) == len(file[self.split]["label"]) == len(file[self.split]["attention_mask"]) == len(file[self.split]["input_ids"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["img_id"])
            self.categories = [category.decode(
                "utf-8") for category in np.unique(file[self.split]["category"])]
            self.num_images = len(np.unique(file[self.split]["img_id"]))
            self.num_labels = json.load(open(self.metadata_file_path, "r"))["num_labels"]
            
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_file_path, 'r')
        output = {}
        output["category"] = self.dataset[self.split + "/category"][idx].decode("utf-8")
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["label"] = self.dataset[self.split + "/label"][idx]
        if self.model_type == "baseline":
            output["pixel_values"] = self.dataset["pixel_values"][output["img_id"]][4]
        elif self.model_type == "patching":
            output["pixel_values"] = self.dataset["pixel_values"][output["img_id"]]
        return output

    def __len__(self):
        return self.dataset_len

class RsvqaBenDataset(torch.utils.data.Dataset):
    def __init__(self, split, model_type):
        self.folder = os.path.join("datasets", "RSVQAxBEN")
        self.dataset = None
        self.split = split
        self.model_type = model_type
        self.image_processing = utils.ImageProcessing(augment_images=False)      
        self.h5_file_name = "rsvqaxben.h5"
        self.h5_file_path = os.path.join(self.folder, self.h5_file_name)
        self.metadata_file_name = "rsvqaxben_metadata.json"
        self.metadata_file_path = os.path.join(self.folder, self.metadata_file_name)
        with h5py.File(os.path.join(self.folder, self.h5_file_name), 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["category"]) == len(file[self.split]["label"]) == len(file[self.split]["attention_mask"]) == len(file[self.split]["input_ids"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["img_id"])
            self.categories = [category.decode(
                "utf-8") for category in np.unique(file[self.split]["category"])]
            self.num_images = len(np.unique(file[self.split]["img_id"]))
            self.num_labels = json.load(open(self.metadata_file_path, "r"))["num_labels"]
    
    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_file_path, 'r')
        output = {}
        output["category"] = self.dataset[self.split + "/category"][idx].decode("utf-8")
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["label"] = self.dataset[self.split + "/label"][idx]
        if self.model_type == "baseline":
            output["pixel_values"] = self.image_processing(Image.open(os.path.join("datasets", "RSVQAxBEN", "images", f"{output['img_id'] // 2000:03d}", str(output["img_id"])+".jpg")))
        elif self.model_type == "patching":
            output["pixel_values"] = self.image_processing(utils.patchImage(os.path.join("datasets", "RSVQAxBEN", "images", f"{output['img_id'] // 2000:03d}", str(output["img_id"])+".jpg")))
        return output

    def __len__(self):
        return self.dataset_len

class NwpuCaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, split, augment_images = False):
        self.folder = os.path.join("datasets", "NWPU-Captions")
        self.dataset = None
        self.split = split
        self.augment_images = augment_images
        self.image_processing = utils.ImageProcessing(augment_images)
        self.h5_file_name = file_name
        self.h5_file_path = os.path.join(self.folder,  self.h5_file_name)
        with h5py.File(os.path.join(self.folder, self.h5_file_name), 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["image"]) == len(file[self.split]["class"]) == len(file[self.split]["caption"]) == len(file[self.split]["filtered_caption"]) == len(
                file[self.split]["sent_id"]) == len(file[self.split]["input_ids"]) == len(file[self.split]["attention_mask"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["sent_id"])
            self.num_classes = len(np.unique(file[self.split]["class"]))
            self.num_images = len(np.unique(file[self.split]["img_id"]))

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_file_path, 'r')
        output = {}
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["image"] = self.dataset[self.split + "/image"][idx].decode("utf-8")
        output["class"] = self.dataset[self.split + "/class"][idx]
        output["caption"] = self.dataset[self.split + "/caption"][idx].decode("utf-8")
        output["filtered_caption"] = self.dataset[self.split + "/filtered_caption"][idx].decode("utf-8")
        output["sent_id"] = self.dataset[self.split + "/sent_id"][idx]
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        if self.augment_images == False:
            output["pixel_values"] = self.dataset[self.split + "/pixel_values"][output["img_id"]]
        else:
            output["pixel_values"] = self.image_processing(Image.open(os.path.join("datasets", "NWPU-Captions", "images", output["img_id"])))
        return output

    def __len__(self):
        return self.dataset_len

class NwpuCaptionsKNN(torch.utils.data.Dataset):
    def __init__(self, split):
        self.folder = os.path.join("datasets", "NWPU-Captions")
        self.dataset = None
        self.split = split
        self.h5_file_name = "nwpu_captions.h5"
        self.h5_file_path = os.path.join(self.folder,  self.h5_file_name)
        with h5py.File(os.path.join(self.folder, self.h5_file_name), 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["image"]) == len(file[self.split]["class"]) == len(file[self.split]["caption"]) == len(file[self.split]["filtered_caption"]) == len(
                file[self.split]["sent_id"]) == len(file[self.split]["input_ids"]) == len(file[self.split]["attention_mask"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(np.unique(file[self.split]["img_id"]))
            self.num_classes = len(np.unique(file[self.split]["class"]))
            self.classes_input_ids = np.array([file["/class_sentence_input_ids"][idx] for idx in range(self.num_classes)])
            self.classes_attention_mask = np.array([file["/class_sentence_attention_mask"][idx] for idx in range(self.num_classes)])

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_file_path, 'r')
        output = {}
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["class"] = self.dataset[self.split + "/class"][idx]
        output["image"] = self.dataset[self.split + "/image"][idx].decode("utf-8")
        output["pixel_values"] = self.dataset["/pixel_values"][output["img_id"]]
        return output

    def __len__(self):
        return self.dataset_len
      