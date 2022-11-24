import torch
import h5py
import numpy as np


class RsvqaDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, mode):
        self.file_path = path
        self.dataset = None
        self.split = split
        self.mode = mode
        with h5py.File(self.file_path, 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["category"]) == len(
                file[self.split]["label"]) == len(file[self.split]["attention_mask"]) == len(file[self.split]["input_ids"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["img_id"])
            self.categories = [category.decode(
                "utf-8") for category in np.unique(file[self.split]["category"])]
            self.num_labels = len(np.unique(file[self.split]["label"]))

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        output = {}
        output["category"] = self.dataset[self.split +
                                          "/category"][idx].decode("utf-8")
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["label"] = self.dataset[self.split + "/label"][idx]
        if self.mode == "baseline":
            output["pixel_values"] = self.dataset["pixel_values"][output["img_id"]][4]
        elif self.mode == "patching":
            output["pixel_values"] = self.dataset["pixel_values"][output["img_id"]]
        return output

    def __len__(self):
        return self.dataset_len


class RsvqaBenDataset(torch.utils.data.Dataset):
    def __init__(self, path, split, mode):
        self.file_path = path
        self.dataset = None
        self.split = split
        self.mode = mode
        with h5py.File(self.file_path, 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["category"]) == len(
                file[self.split]["label"]) == len(file[self.split]["attention_mask"]) == len(file[self.split]["input_ids"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["img_id"])
            self.categories = [category.decode(
                "utf-8") for category in np.unique(file[self.split]["category"])]
            self.num_labels = len(np.unique(file[self.split]["label"]))

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        output = {}
        output["category"] = self.dataset[self.split +
                                          "/category"][idx].decode("utf-8")
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        output["img_id"] = self.dataset[self.split + "/img_id"][idx]
        output["label"] = self.dataset[self.split + "/label"][idx]
        if self.mode == "baseline":
            # TODO fazer processamento da imagem aqui
            output["pixel_values"] = output["img_id"]
        elif self.mode == "patching":
            # TODO fazer processamento da imagem aqui
            output["pixel_values"] = output["img_id"]
        return output

    def __len__(self):
        return self.dataset_len


class NwpuCaptionsDataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        self.file_path = path
        self.dataset = None
        self.split = split
        with h5py.File(self.file_path, 'r') as file:
            assert len(file[self.split]["img_id"]) == len(file[self.split]["class"]) == len(file[self.split]["caption"]) == len(file[self.split]["sent_id"]) == len(
                file[self.split]["input_ids"]) == len(file[self.split]["attention_mask"]), "non matching number of entries in .h5 file."
            self.dataset_len = len(file[self.split]["img_id"])
            self.num_classes = len(np.unique(file[self.split]["class"]))
            self.num_images = len(np.unique(file[self.split]["img_id"]))

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        output = {}
        output["img_id"] = self.dataset[self.split +
                                        "/img_id"][idx].decode("utf-8")
        output["class"] = self.dataset[self.split + "/class"][idx]
        output["caption"] = self.dataset[self.split +
                                          "/caption"][idx].decode("utf-8")
        output["filtered_caption"] = self.dataset[self.split +
                                          "/filtered_caption"][idx].decode("utf-8")
        output["sent_id"] = self.dataset[self.split + "/sent_id"][idx]
        output["input_ids"] = self.dataset[self.split + "/input_ids"][idx]
        output["attention_mask"] = self.dataset[self.split + "/attention_mask"][idx]
        return output

    def __len__(self):
        return self.dataset_len
