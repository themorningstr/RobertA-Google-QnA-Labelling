import Config
import torch
import torch.nn as nn


class GoogleQnADataset(nn.Module):
    def __init__(self,qTitle,qBody,answer,targets):
        self.qTitle = qTitle
        self.qBody = qBody
        self.answer = answer
        self.targets = targets
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN

    def __len__(self):
        return len(self.qTitle)

    def __getitem__(self,item):
        question_title = str(self.qTitle[item])
        question_body = str(self.qBody[item])
        answer = str(self.answer[item])

        # [CLS] [Q-TITLE] [Q-BODY] [SEP] [ANSWER] [SEP]

        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer,
            add_special_token = True,
            max_len = self.max_len,
            truncation = True
            )

        ids = inputs["input_ids"]
        masks = inputs["attention_mask"]



        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        masks = masks + ([0] * padding_length)

        return {
            "ids" : torch.tensor(ids,dtype = torch.long),
            "masks" : torch.tensor(masks,dtype = torch.long),
            "targets" : torch.tensor(self.targets[item, :],dtype = torch.float)
        }
