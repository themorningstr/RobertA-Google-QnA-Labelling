import torch
import Dataset
import Config
import Engine
import Model
import Utils
import pandas as pd
import numpy as np
import transformers
from sklearn import model_selection
from tqdm import trange




def train():
    # Loading the Data

    df1 = pd.read_csv("Data/train.csv")
    df2 = pd.read_csv("Data/sample_submission.csv")

    # defining the target columns
    target_cols = df2.columns[1:]

    # Spliting the data into Train and Validation

    df_train ,df_valid = model_selection.train_test_split(
        df1,
        test_size = 0.2,
        random_state = 2021,
        )


    # Creating Training and Validation Dataset

    Train_Dataset = Dataset.GoogleQnADataset(
        qTitle = df_train.question_title.values,
        qBody = df_train.question_body.values,
        answer = df_train.answer.values,
        targets = df_train[target_cols].values)


    Valid_Dataset = Dataset.GoogleQnADataset(
        qTitle = df_valid.question_title.values,
        qBody = df_valid.question_body.values,
        answer = df_valid.answer.values,
        targets = df_valid[target_cols].values)

    # Initilization of Train DataLoader and Validation DataLoader

    Train_DataLoader = torch.utils.data.DataLoader(
        Train_Dataset,
        batch_size =  Config.TRAIN_BATCH_SIZE,
        sampler = torch.utils.data.RandomSampler(Train_Dataset)
    )

    Valid_DataLoader = torch.utils.data.DataLoader(
        Valid_Dataset,
        batch_size = Config.VALID_BATCH_SIZE,
        sampler = torch.utils.data.SequentialSampler(Valid_Dataset)
    )

    # Initilizing the Model
    config = transformers.RobertaConfig.from_pretrained(Config.MODEL_BASE_PATH)
    model = Model.ROBERTAModel(
        conf = config,
        num_label = Config.NUMBER_OF_LABEL
        )
    
    # Initilizing the optimizer 

    optimizer_grouped_parameters = Utils.optimizer_params(Model = model)

    optimizer = transformers.AdamW(
        optimizer_grouped_parameters, 
        lr = 3e-5,
        correct_bias = True)

    # Initilizing the Scheduler

    total_steps = int(len(df1) / Config.TRAIN_BATCH_SIZE * Config.EPOCH)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    for epoch in trange(Config.EPOCH, desc = "EPOCH"):
        Engine.Train(
            DataLoader = Train_DataLoader,
            Model = model, 
            Optimizer = optimizer,
            Device=Config.DEVICE,
            Scheduler=scheduler
        )

        valid_loss,targets,output = Engine.Eval(
            DataLoader = Valid_DataLoader,
            Model=model,
            Device = Config.DEVICE
        )

    best_loss = np.inf
    if valid_loss < best_loss:
        torch.save(model.state_dict(),"model.bin")
        valid_loss = best_loss



if __name__ == "__main__":
    train()