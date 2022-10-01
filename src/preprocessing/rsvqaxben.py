import os

import datasets
import images

benDataset = datasets.load_dataset("csv", data_files={"train": os.path.join("datasets", "RSVQAxBEN", "traindf.csv"),
                                                      "test": os.path.join("datasets", "RSVQAxBEN", "testdf.csv"),
                                                      "validation": os.path.join("datasets", "RSVQAxBEN", "valdf.csv")})
benDataset = datasets.DatasetDict({"train": benDataset["train"],
                                   "test": benDataset["test"],
                                   "validation": benDataset["validation"]})

benDataset = benDataset.remove_columns(["mode"])

print(benDataset)
print("saving to disk...")

benDataset.save_to_disk(os.path.join("datasets", "RSVQAxBEN", "dataset"))

images.jpgOnly("RSVQAxBEN")
#images.imageResizer("RSVQAxBEN")
#images.verifyImages()
