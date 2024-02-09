import os
import torch


# general
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD = "<__PAD__>"
UNK = "<__UNK__>"
NOFIX = "<__NOFIX__>"
SOS = "<__SOS__>"
EOS = "<__EOS__>"

batch_size = 1
teacher_forcing_ratio = 0.5
embedding_dim = 300
fix_hidden_dim = 128
sem_hidden_dim = 1024
fix_dropout = 0.5
par_dropout = 0.2
_fix_learning_rate = 0.00001
_par_learning_rate = 0.0001
learning_rate = _par_learning_rate 
fix_momentum = 0.9
par_momentum = 0.0
max_length = 851
epochs = 5

# paths
data_path = "./data"

emb_path = os.path.join(data_path, "Google_word2vec/GoogleNews-vectors-negative300.bin")

glove_path = '/kaggle/input/glove-sent/glove.840B.300d.txt'

google_path = os.path.join(data_path, "datasets/sentence-compression/data")
google_train_path = '/kaggle/input/sentence-compression/0000.parquet'
google_dev_path = '/kaggle/input/sentence-compression/0000 (2).parquet'
google_test_path = '/kaggle/input/sentence-compression/0002.parquet'
