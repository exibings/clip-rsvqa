from transformers import CLIPModel, CLIPConfig


config = CLIPConfig.from_pretrained("saved-models/clip-rs")
#print(config)
model = CLIPModel.from_pretrained("saved-models/clip-rs")
print(model)