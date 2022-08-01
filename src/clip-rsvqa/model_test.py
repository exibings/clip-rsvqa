import torch
import os
import datasets
from Model import CLIPxRSVQA
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_folder = os.path.join("datasets", "RSVQA-LR", "images")
dataset = datasets.load_from_disk(os.path.join("datasets", "RSVQA-LR", "dataset"))

# encode labels
labels = list(set(dataset["train"]["answer"]))
label2id = {}
id2label = {}
count = 0
for label in labels:
    label2id[label] = count
    id2label[count] = label
    count += 1

input_processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")

model = CLIPxRSVQA(config=clip_model.config, num_labels=len(label2id), device=device)
model.text_model = clip_model.text_model
model.vision_model = clip_model.vision_model
model.visual_projection = clip_model.visual_projection
model.text_projection = clip_model.text_projection
model.logit_scale = clip_model.logit_scale


model.to(device)  # send model to GPU
dataset_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=64,
                                             shuffle=True, num_workers=2)


def prepareBatch(batch: dict) -> dict:
    """
    Prepares batch for model training. Sends batch to GPU. Returns the processed batch.

    Args:
        batch (dict): Batch given by torch.utils.data.DataLoader.

    Returns:
        dict: processed batch in GPU, ready to be fed to model.
    """
    # create batch
    img_paths = []

    for img_id in batch["img_id"].tolist():
        if os.path.exists(os.path.join(img_folder, str(img_id) + ".jpg")):
            img_paths.append(os.path.join(img_folder, str(img_id) + ".jpg"))

    imgs_to_encode = [Image.open(img) for img in img_paths]

    # process the entire batch at once with padding for dynamic padding
    processed_batch = input_processor(
        text=batch["question"], images=imgs_to_encode, padding=True, return_tensors="pt")
    del imgs_to_encode  # free up memory from imgs
    processed_input = {**{"labels": torch.tensor([label2id[label]
                                                  for label in batch["answer"]])}, **dict(processed_batch)}

    # send tensors to GPU
    for key in processed_input:
        processed_input[key] = processed_input[key].to(device)
    return processed_input


model(**prepareBatch(next(iter(dataset_loader))))
