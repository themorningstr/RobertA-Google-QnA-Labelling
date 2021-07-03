import torch
import numpy as np



def Loss(output,target):
    return torch.nn.BCEWithLogitsLoss()(output,target)
    


def Train(DataLoader,Model,Optimizer,Device,Scheduler = None):
    Model.train()

    for index,batch in enumerate(DataLoader):
        ids = batch["ids"]
        masks = batch["masks"]
        target = batch["targets"]

        ids = ids.to(Device,dtype = torch.long)
        masks = masks.to(Device,dtype = torch.long)
        target = target.to(Device,dtype = torch.float)

        Optimizer.zero_grad()

        output = Model(
            input_ids = ids,
            attention_mask = masks
        )
        loss = Loss(output,target)
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.0)

        Optimizer.step()

        if Scheduler is not None:
            Scheduler.step()

        if index % 10 == 0:
            print(f"Index {index} >>>====================>>> Train Loss {loss}")



def Eval(DataLoader,Model,Device):
    Model.eval()
    final_outputs = []
    final_targets = []

    for index,batch in enumerate(DataLoader):

        ids = batch["ids"]
        masks = batch["masks"]
        target = batch["targets"]


        ids = ids.to(Device,dtype = torch.long)
        masks = masks.to(Device,dtype = torch.long)
        target = target.to(Device,dtype = torch.float)


        output = Model(
            input_ids = ids,
            attention_mask = masks
        )

        loss = Loss(output,target)

        final_targets.extend(target.cpu().detach().numpy().tolist())
        final_outputs.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())

        if index % 10 == 0:
            print(f"Index : {index} >>>====================>>> Valid Loss : {loss}")
    
    return loss,np.vstack(final_targets), np.vstack(final_outputs)

        

