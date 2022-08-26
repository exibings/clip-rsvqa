import os

import datasets
import images

hrDataset = datasets.load_dataset("csv", data_files={"train": os.path.join("datasets", "RSVQA-HR", "traindf.csv"),
                                                     "test": os.path.join("datasets", "RSVQA-HR", "testdf.csv"),
                                                     "test_phili": os.path.join("datasets", "RSVQA-HR", "testdf_phili.csv"),
                                                     "validation": os.path.join("datasets", "RSVQA-HR", "testdf.csv")})
hrDataset = datasets.DatasetDict({"train": hrDataset["train"],
                                  "test": hrDataset["test"],
                                  "test_phili": hrDataset["test_phili"],
                                  "validation": hrDataset["validation"]})

hrDataset = hrDataset.remove_columns(["mode"])

print(hrDataset)
print("saving to disk...")

hrDataset.save_to_disk(os.path.join("datasets", "RSVQA-HR", "dataset"))

images.jpgOnly("RSVQA-HR")
images.imageResizer("RSVQA-HR")
images.verifyImages()
