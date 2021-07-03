import torch
import Config
import transformers



class ROBERTAModel(torch.nn.Module):
    def __init__(self,conf,num_label):
        super(ROBERTAModel,self).__init__()
        self.conf = conf
        self.num_label = num_label
        self.roberta = transformers.RobertaModel(self.conf,add_pooling_layer = False)
        self.dense = torch.nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE)
        self.dropout = torch.nn.Dropout(Config.HIDDEN_DROPOUT_PROB)
        self.classifier = torch.nn.Linear(Config.HIDDEN_SIZE, self.num_label)

    def forward(self,input_ids,attention_mask):
        output,_ = self.roberta(input_ids,attention_mask,return_dict = False)
        output = output[:,0,:]
        output = self.dropout(output)
        output = self.dense(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        output = self.classifier(output)
        return output

     
