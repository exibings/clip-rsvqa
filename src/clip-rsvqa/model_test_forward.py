from concurrent.futures import process
import torch
import os
import datasets
from Model import CLIPxRSVQA
from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPProcessor
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_folder = os.path.join("datasets", "RSVQA-LR", "images")
dataset = datasets.load_from_disk(os.path.join("datasets", "RSVQA-LR", "dataset"))

# encode labels
labels = set(dataset["train"]["answer"]).union(dataset["validation"]["answer"]).union(dataset["test"]["answer"])
label2id = {}
id2label = {}
count = 0
for label in labels:
    label2id[label] = count
    id2label[count] = label
    count += 1
input_processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
feature_extractor = CLIPFeatureExtractor.from_pretrained("flax-community/clip-rsicd-v2")
tokenizer = CLIPTokenizer.from_pretrained("flax-community/clip-rsicd-v2")

model = CLIPxRSVQA(num_labels=len(label2id))
model.to(device)  # send model to GPU
print(model)
dataset_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=2,
                                             shuffle=False, num_workers=2)


def patchImage(img_path):
    img = Image.open(img_path)
    return [img.crop((0, 0, img.width//2, img.height//2)), img.crop((img.width//2, 0, img.width, img.height//2)),
            img.crop((0, img.height//2, img.width//2, img.height)), img.crop((img.width//2, img.height//2, img.width, img.height)), img]


def prepareBatch(batch: dict) -> dict:
    """
    Prepares batch to feed the model. Sends batch to GPU. Returns the processed batch.

    Args:
        batch (dict): Dataset batch given by torch.utils.data.DataLoader

    Returns:
        dict: {"labels": tensor with #batch_size elements,
                "pixel_values": tensor with the pixel values for all the images and respective patches needed for the batch. [n_patches+1=5, batch_size, n_channels=3, height=224, width=224],
                "input_ids": tesnor with the encoded questions. #batch_size elements,
                "attention_mask": tensor with the attention masks for each question. #batch_size elements}
    """
    # create training batch
    processed_input = {}
    processed_imgs = {}
    for img_id in batch["img_id"].tolist():
        if os.path.exists(os.path.join(img_folder, str(img_id) + ".jpg")):
            if img_id not in processed_imgs:
                processed_imgs[img_id] = feature_extractor(patchImage(os.path.join(img_folder, str(
                    img_id) + ".jpg")), return_tensors="pt", resample=Image.Resampling.BILINEAR)
                # print(processed_imgs[img_id])
            if "pixel_values" not in processed_input:
                processed_input["pixel_values"] = processed_imgs[img_id]["pixel_values"]
            else:
                processed_input["pixel_values"] = torch.stack(
                    (processed_input["pixel_values"], processed_imgs[img_id]["pixel_values"]), dim=1)

    processed_input.update({"labels": torch.tensor([label2id[label] for label in batch["answer"]]),
                            **tokenizer(batch["question"], padding=True, return_tensors="pt")})
    # send tensors to GPU
    for key in processed_input:
        processed_input[key] = processed_input[key].to(device)
    return processed_input


processed_input = prepareBatch(next(iter(dataset_loader)))
#print("pixel values size", processed_input["pixel_values"].size())
#print("patch1 size", processed_input["pixel_values"][0].size())
#print("patch1", processed_input["pixel_values"][0])
#print("patch2 size", processed_input["pixel_values"][1].size())
#print("patch2", processed_input["pixel_values"][1])
#print("patch3 size", processed_input["pixel_values"][2].size())
#print("patch3", processed_input["pixel_values"][2])
#print("patch4 size", processed_input["pixel_values"][3].size())
#print("patch4", processed_input["pixel_values"][3])
#print("full image size", processed_input["pixel_values"][4].size())
#print("full image", processed_input["pixel_values"][4])
model(**processed_input)
