
import os
import datasets
from PIL import Image

benDataset = datasets.load_dataset("csv", data_files={"train": os.path.join("datasets", "RSVQAxBEN", "traindf.csv"), 
                                                         "test": os.path.join("datasets", "RSVQAxBEN", "testdf.csv"), 
                                                         "validation": os.path.join("datasets", "RSVQAxBEN", "testdf.csv")})
benDataset = datasets.DatasetDict({"train": benDataset["train"],
                                      "test": benDataset["test"],
                                      "validation":benDataset["validation"]})

benDataset = benDataset.remove_columns(["mode", "category"])

print(benDataset)
print("saving to disk...")

benDataset.save_to_disk(os.path.join("datasets", "RSVQAxBEN","dataset"))