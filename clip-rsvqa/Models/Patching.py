import torch
from transformers import CLIPModel
from copy import deepcopy


class CLIPxRSVQA(CLIPModel):
    def __init__(self, num_labels, model_aspect_ratio: tuple = (3,8)):
        clip_model = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
        super().__init__(clip_model.config)
        self.text_model = clip_model.text_model
        self.increase_position_embeddings(128)
        self.aspect_ratio = model_aspect_ratio
        self.vision_model = clip_model.vision_model
        self.full_image_projection = clip_model.visual_projection
        self.patching_projection1 = deepcopy(clip_model.visual_projection)
        self.patching_projection2 = deepcopy(clip_model.visual_projection)
        self.patching_projection3 = deepcopy(clip_model.visual_projection)
        self.patching_projection4 = deepcopy(clip_model.visual_projection)
        self.text_projection = clip_model.text_projection
        self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=model_aspect_ratio[0])
        self.new_transformer_encoder = torch.nn.TransformerEncoder(
            self.new_encoder_layer, num_layers=model_aspect_ratio[1])
        self.classification = torch.nn.Linear(512, num_labels, bias=True)
        self.logit_scale = clip_model.logit_scale
        self.num_labels = num_labels

    def forward(self, input_ids=None, pixel_values: dict = None, attention_mask=None, position_ids=None, labels=None):
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
        image_embeds = self.image_fusing(pixel_values)  # size: [batch_size, 50*#patches, 512]
        # size: [batch_size, 50*#patches+sequence_length, 512]
        joint_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        # size: [batch_size, 50*#patches+sequence_length], where the first 50*#patches elements are ones - images are not masked
        joint_embeds_mask = torch.cat(
            (torch.ones((image_embeds.size()[0], image_embeds.size()[1])).to(device=self.device), attention_mask), dim=1)
        # src size: [50*#patches+sequence_length, batch_size, 512]; src_key_padding_mask size: [batch_size, 50*#patches]
        multimodal_embed = self.new_transformer_encoder(src=joint_embeds.permute(1, 0, 2),
                                                        src_key_padding_mask=joint_embeds_mask).permute(1, 0, 2)  # size: [batch_size, 50*#patches+sequence_length, 512]
        multimodal_embed_mask = self.create_multimodal_embed_mask(
            text_attention_mask=attention_mask, image_embeds=image_embeds, text_embeds=text_embeds)  # size: [batch_size, 50*#patches+sequence_length, 512]

        multimodal_embed = self.mean_pooling(multimodal_embed, multimodal_embed_mask)  # size: [batch_size, 512]
        return self.classification(multimodal_embed)  # size: [batch_size, n_labels]

    def image_fusing(self, pixel_values):
        patch1 = self.vision_model(
            pixel_values=pixel_values[:, 0],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch1_embeds = self.patching_projection1(patch1.last_hidden_state)
        patch1_embeds = patch1_embeds / patch1_embeds.norm(p=2, dim=-1, keepdim=True)

        patch2 = self.vision_model(
            pixel_values=pixel_values[:, 1],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch2_embeds = self.patching_projection2(patch2.last_hidden_state)
        patch2_embeds = patch2_embeds / patch2_embeds.norm(p=2, dim=-1, keepdim=True)

        patch3 = self.vision_model(
            pixel_values=pixel_values[:, 2],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch3_embeds = self.patching_projection3(patch3.last_hidden_state)
        patch3_embeds = patch3_embeds / patch3_embeds.norm(p=2, dim=-1, keepdim=True)

        patch4 = self.vision_model(
            pixel_values=pixel_values[:, 3],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        patch4_embeds = self.patching_projection4(patch4.last_hidden_state)
        patch4_embeds = patch4_embeds / patch4_embeds.norm(p=2, dim=-1, keepdim=True)

        full_image = self.vision_model(
            pixel_values=pixel_values[:, 4],
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        full_image_embeds = self.full_image_projection(full_image.last_hidden_state)
        full_image_embeds = full_image_embeds / full_image_embeds.norm(p=2, dim=-1, keepdim=True)

        return torch.cat((patch1_embeds, patch2_embeds, patch3_embeds, patch4_embeds, full_image_embeds), dim=1)

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

    def increase_position_embeddings(self, new_max):
        #print("before")
        #print("size:", self.text_model.embeddings.position_embedding.weight.shape)
        #print("values[0:77]:", self.text_model.embeddings.position_embedding.weight[0:77])
        #print("values[54:100]:", self.text_model.embeddings.position_embedding.weight[54:77])
        current_max_pos, embed_size = self.text_model.embeddings.position_embedding.weight.shape
        assert new_max > current_max_pos
        # allocate a larger position embedding matrix
        new_pos_embed = self.text_model.embeddings.position_embedding.weight.new_empty(new_max, embed_size)
        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        while k+current_max_pos <= new_max:
            new_pos_embed[k:(k + current_max_pos)] = self.text_model.embeddings.position_embedding.weight
            k += current_max_pos
        new_pos_embed[k:new_max] = self.text_model.embeddings.position_embedding.weight[current_max_pos - (new_max - k):current_max_pos]
        self.text_model.embeddings.position_embedding.weight.data = new_pos_embed
        self.text_model.embeddings.position_ids.data = torch.tensor([i for i in range(new_max)]).reshape(1, new_max)
        #print("after")
        #print("size:", self.text_model.embeddings.position_embedding.weight.shape)
        #print("values[0:77]:", self.text_model.embeddings.position_embedding.weight[0:77])
        #print("values[77:100]:", self.text_model.embeddings.position_embedding.weight[77:100])