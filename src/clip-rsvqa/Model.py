import torch
from torch.nn import CrossEntropyLoss
from transformers import CLIPModel


class CLIPxRSVQA(CLIPModel):
    def __init__(self, config, num_labels, device):
        super().__init__(config)  # este config é o clip_model.config que está no training.py
        self.new_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.new_transformer_encoder = torch.nn.TransformerEncoder(self.new_encoder_layer, num_layers=3)
        self.classification = torch.nn.Linear(512, num_labels, bias=True)
        self.num_labels = num_labels
        self.device = device

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, return_loss=None, output_attentions=None, output_hidden_states=None, labels=None):
        output = super().forward(input_ids, pixel_values, attention_mask, position_ids,
                                 return_loss, output_attentions, output_hidden_states, return_dict=True)

        aux_vision = output.vision_model_output[0]
        aux_vision = self.visual_projection(aux_vision)
        aux_text = output.text_model_output[0]
        aux_text = self.text_projection(aux_text)

        #print("input_ids:", input_ids, "inputs_id size:", input_ids.size())
        #print("aux_text:", aux_text, "aux_text size:", aux_text.size())
        #print("vision projection size:", aux_vision.size(), "/ text projection size:", aux_text.size())

        aux = torch.cat((aux_vision, aux_text), dim=1)
        #print("initial multi modal tensor size:", aux.size())
        #print("multi modal tensor size needs to be (sequence length, number of batches, feature number)", "(", aux.size()[1], ",", aux.size()[0], ",", aux.size()[2] , ")")
        aux = aux.reshape((aux.size()[1], aux.size()[0], aux.size()[2]))
        #print("after reshape multi modal tensor:", aux, "multi modal tensor size:", aux.size())
        vision_mask = torch.ones((aux_vision.size()[0], aux_vision.size()[1])).to(self.device)
        #print("text mask", attention_mask, "text size:", attention_mask.size())
        #print("vision mask:", vision_mask, "vision mask size:", vision_mask.size())

        #print("text_projection_mask:", text_projection_mask, "text_projection_mask size:", text_projection_mask.size())
        multi_modal_mask = torch.cat((vision_mask, attention_mask), dim=1).to(self.device)

        #print("multi_modal_mask tensor:", multi_modal_mask, "multi_modal_mask size:", multi_modal_mask.size())
        aux = self.new_transformer_encoder(aux, src_key_padding_mask=multi_modal_mask)
        # change back shape to (batch size, sequence length, features)
        aux = aux.reshape((aux.size()[1], aux.size()[0], aux.size()[2]))
        #print("trasnformer encoder output:", aux, "transformer encoder output size:", aux.size())

        multi_modal_mask = multi_modal_mask.unsqueeze(2).expand(-1, -1, aux.size()[2])

        # TODO experimentar a mask inicial e ver se ele faz as contas bem na mesma, se nao deixar ficar assim
        #print("expanded multi_modal_mask tensor:", multi_modal_mask, "expanded multi_modal_mask size:", multi_modal_mask.size())
        aux = torch.sum(aux * multi_modal_mask, 1) / torch.clamp(multi_modal_mask.sum(1), min=1e-9)
        aux = self.classification(aux)

        #print("classification:", aux, "classification size:", aux.size())
        output.logits = aux
        output.loss = None
        #print("labels:", labels)
        #print("forward output with no loss:", output)
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"  # cenários com várias respostas possíveis
            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(output.logits.squeeze(), labels.squeeze())
                else:
                    output.loss = loss_fct(output.logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                output.loss = loss_fct(output.logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                output.loss = loss_fct(output.logits, labels)
        #print("problem type:", self.config.problem_type)
        return output
