import torch
from torch.nn import CrossEntropyLoss
from transformers import CLIPModel
import copy


class CLIPxRSVQA(CLIPModel):
    def __init__(self, num_labels):
        clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        super().__init__(clip_model.config)
        self.text_model = clip_model.text_model
        self.vision_model = clip_model.vision_model
        self.patching_projection1 = clip_model.visual_projection
        self.patching_projection2 = clip_model.visual_projection
        self.patching_projection3 = clip_model.visual_projection
        self.patching_projection4 = clip_model.visual_projection
        self.full_image_projection = clip_model.visual_projection
        self.text_projection = clip_model.text_projection
        self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.new_transformer_encoder = torch.nn.TransformerEncoder(self.new_encoder_layer, num_layers=3)
        self.classification = torch.nn.Linear(512, num_labels, bias=True)
        self.logit_scale = clip_model.logit_scale
        self.loss = CrossEntropyLoss()
        self.num_labels = num_labels
#        print("patch1 layer weights:", self.patching_projection1.weight)
#        print("patch2 layer weights:", self.patching_projection2.weight)
#        print("patch3 layer weights:", self.patching_projection3.weight)
#        print("patch4 layer weights:", self.patching_projection4.weight)
#        print("full image layer weights:", self.full_image_projection.weight)

    def forward(self, input_ids=None, pixel_values: dict = None, attention_mask=None, position_ids=None, labels=None):
        outputs = {}

        # quando for os 4 patches apenas chamar 4 vezes o CLIP image
        patch1 = self.vision_model(
            pixel_values=pixel_values[0],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch1_embeds = self.patching_projection1(patch1.last_hidden_state)
        print("patch1_embeds.size()", patch1_embeds.size())

        patch2 = self.vision_model(
            pixel_values=pixel_values[1],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch2_embeds = self.patching_projection2(patch2.last_hidden_state)
        print("patch2_embeds.size()", patch2_embeds.size())

        patch3 = self.vision_model(
            pixel_values=pixel_values[2],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch3_embeds = self.patching_projection3(patch3.last_hidden_state)
        print("patch3_embeds.size()", patch3_embeds.size())

        patch4 = self.vision_model(
            pixel_values=pixel_values[3],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch4_embeds = self.patching_projection4(patch4.last_hidden_state)
        print("patch4_embeds.size()", patch4_embeds.size())

        full_image = self.vision_model(
            pixel_values=pixel_values[4],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        full_image_embeds = self.full_image_projection(full_image.last_hidden_state)
        print("full_image_embeds.size()", full_image_embeds.size())

        image_embeds = torch.stack((patch1_embeds, patch2_embeds, patch3_embeds, patch4_embeds, full_image_embeds))
        print("image_embeds.size()", image_embeds.size())  # size: [5*batch_size, 50, 512]

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        text_embeds = text_outputs.last_hidden_state  # text_outputs['last_hidden_state']
        text_embeds = self.text_projection(text_embeds)  # size: [batch_size, sequence_length, 512]
        # normalize features
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return

        joint_embeds = torch.cat((image_embeds, text_embeds), dim=1)  # size: [batch_size, 50+sequence_length, 512]
        # size: [batch_size, 50+sequence_length], where the first 50 elements are ones - images are not masked
        joint_embeds_mask = torch.cat(
            (torch.ones((image_embeds.size()[0], image_embeds.size()[1])).to(device=self.device), attention_mask), dim=1)
        # src size: [50+sequence_length, batch_size, 512]; src_key_padding_mask size: [batch_size, 50+sequence_length]
        multimodal_embed = self.new_transformer_encoder(src=joint_embeds.permute(1, 0, 2),
                                                        src_key_padding_mask=joint_embeds_mask).permute(1, 0, 2)  # size: [batch_size, 50+sequence_length, 512]
        multimodal_embed_mask = self.create_multimodal_embed_mask(
            text_attention_mask=attention_mask, image_embeds=image_embeds, text_embeds=text_embeds)  # size: [batch_size, 50+sequence_length, 512]

        multimodal_embed = self.mean_pooling(multimodal_embed, multimodal_embed_mask)  # size: [batch_size, 512]
        outputs["logits"] = self.classification(multimodal_embed)  # size: [batch_size, n_labels]
        outputs["loss"] = self.loss(outputs["logits"], labels)
        return outputs

    def mean_pooling(self, multimodal_embed, multimodal_mask=None):
        if multimodal_mask == None:
            return multimodal_embed.mean(dim=1)
        else:
            return torch.sum(multimodal_embed * multimodal_mask, 1) / torch.clamp(multimodal_mask.sum(1), min=1e-9)

    def create_multimodal_embed_mask(self, text_attention_mask, image_embeds, text_embeds):
        image_embeds_mask = torch.ones(image_embeds.size()).to(self.device)
        text_embeds_mask = text_attention_mask.unsqueeze(-1).expand(text_embeds.size())
        multimodal_embed_mask = torch.cat((image_embeds_mask, text_embeds_mask),
                                          dim=1)  # size: [batch_size, 50+sequence_length, 512]
        return multimodal_embed_mask

    def freeze_vision(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
