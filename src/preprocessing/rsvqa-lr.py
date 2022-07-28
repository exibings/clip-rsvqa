
import os

import datasets

lrDataset = datasets.load_dataset("csv", data_files={"train": os.path.join("datasets", "RSVQA-LR", "traindf.csv"), 
                                                         "test": os.path.join("datasets", "RSVQA-LR", "testdf.csv"), 
                                                         "validation": os.path.join( "datasets", "RSVQA-LR", "testdf.csv")})
lrDataset = datasets.DatasetDict({"train": lrDataset["train"],
                                      "test": lrDataset["test"],
                                      "validation":lrDataset["validation"]})

lrDataset = lrDataset.remove_columns(["mode", "category"])

print(lrDataset)
print("saving to disk...")

lrDataset.save_to_disk(os.path.join("datasets", "RSVQA-LR","dataset"))