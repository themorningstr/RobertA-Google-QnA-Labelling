import torch
import transformers


EPOCH = 5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_BASE_PATH = "roberta-base"
HIDDEN_SIZE = 512
NUMBER_OF_LABEL = 30
HIDDEN_DROPOUT_PROB = 0.3
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TOKENIZER = transformers.RobertaTokenizer.from_pretrained(MODEL_BASE_PATH,do_lower_case = True)