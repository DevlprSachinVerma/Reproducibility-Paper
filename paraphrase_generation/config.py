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
embedding_dim = 768
fix_hidden_dim = 128
par_hidden_dim = 1024
fix_dropout = 0.5
par_dropout = 0.2
_fix_learning_rate = 0.00001
_par_learning_rate = 0.0001
learning_rate = _par_learning_rate
fix_momentum = 0.9
par_momentum = 0.0
max_length = 121
epochs = 5

# paths
data_path = "./data"
provo_predictability_path = os.path.join(
    data_path, "datasets/provo/Provo_Corpus-Predictability_Norms.csv"
)
provo_eyetracking_path = os.path.join(
    data_path, "datasets/provo/Provo_Corpus-Eyetracking_Data.csv"
)

geco_en_path = os.path.join(data_path, "datasets/geco/EnglishMaterial.csv")
geco_mono_path = os.path.join(data_path, "datasets/geco/MonolingualReadingData.csv")

movieqa_human_path = os.path.join(data_path, "datasets/all_word_scores_fixations")
movieqa_human_path_2 = os.path.join(
    data_path, "datasets/all_word_scores_fixations_exp2"
)
movieqa_human_path_3 = os.path.join(
    data_path, "datasets/all_word_scores_fixations_exp3"
)
movieqa_split_plot_path = os.path.join(data_path, "datasets/split_plot_UNRESOLVED")

cnn_path = os.path.join(
    data_path,
    "projects/2019/fixation_prediction/ez-reader-wrapper/predictability/output_cnn/",
)
dm_path = os.path.join(
    data_path,
    "projects/2019/fixation_prediction/ez-reader-wrapper/predictability/output_dm/",
)

qqp_paws_basedir = os.path.join(data_path, "datasets/paw_google/qqp/paws_qqp/output")
qqp_paws_train_path = os.path.join(qqp_paws_basedir, "train.tsv")
qqp_paws_dev_path = os.path.join(qqp_paws_basedir, "dev.tsv")
qqp_paws_test_path = os.path.join(qqp_paws_basedir, "test.tsv")

qqp_basedir =  "/kaggle/input/supporting-files/quora_duplicate_questions.tsv"
qqp_train_path = "/kaggle/input/dataset/train.tsv"
qqp_dev_path = "/kaggle/input/dataset/validation.tsv"
qqp_test_path = "/kaggle/input/dataset/test.tsv"

qqp_kag_basedir = os.path.join(data_path, "datasets/Quora_question_pair_partition_kag")
qqp_kag_train_path = os.path.join(qqp_kag_basedir, "train.tsv")
qqp_kag_dev_path = os.path.join(qqp_kag_basedir, "dev.tsv")
qqp_kag_test_path = os.path.join(qqp_kag_basedir, "test.tsv")

wiki_basedir = os.path.join(data_path, "datasets/paw_google/wiki")
wiki_train_path = os.path.join(wiki_basedir, "train.tsv")
wiki_dev_path = os.path.join(wiki_basedir, "dev.tsv")
wiki_test_path = os.path.join(wiki_basedir, "test.tsv")

msrpc_basedir = os.path.join(data_path, "datasets/MSRPC")
msrpc_train_path = os.path.join(msrpc_basedir, "msr_paraphrase_train.txt")
msrpc_dev_path = os.path.join(msrpc_basedir, "msr_paraphrase_dev.txt")
msrpc_test_path = os.path.join(msrpc_basedir, "msr_paraphrase_test.txt")

sentiment_basedir = os.path.join(data_path, "datasets/sentiment_kag")
sentiment_train_path = os.path.join(sentiment_basedir, "train.tsv")
sentiment_dev_path = os.path.join(sentiment_basedir, "dev.tsv")
sentiment_test_path = os.path.join(sentiment_basedir, "test.tsv")

tamil_basedir = os.path.join(data_path, "datasets/en-ta-parallel-v2")
tamil_train_path = os.path.join(tamil_basedir, "corpus.bcn.train.enta")
tamil_dev_path = os.path.join(tamil_basedir, "corpus.bcn.dev.enta")
tamil_test_path = os.path.join(tamil_basedir, "corpus.bcn.test.enta")

compression_basedir = os.path.join(data_path, "datasets/sentence-compression/data")
compression_train_path = os.path.join(compression_basedir, "train.tsv")
compression_dev_path = os.path.join(compression_basedir, "dev.tsv")
compression_test_path = os.path.join(compression_basedir, "test.tsv")

stanford_basedir = os.path.join(data_path, "datasets/stanfordSentimentTreebank")
stanford_train_path = os.path.join(stanford_basedir, "train.tsv")
stanford_dev_path = os.path.join(stanford_basedir, "dev.tsv")
stanford_test_path = os.path.join(stanford_basedir, "test.tsv")

stanford_sent_basedir = os.path.join(data_path, "datasets/stanfordSentimentTreebank")
stanford_sent_train_path = os.path.join(stanford_basedir, "train_sent.tsv")
stanford_sent_dev_path = os.path.join(stanford_basedir, "dev_sent.tsv")
stanford_sent_test_path = os.path.join(stanford_basedir, "test_sent.tsv")


emb_path = os.path.join(data_path, "Google_word2vec/GoogleNews-vectors-negative300.bin")

glove_path = "/kaggle/input/dataset/glove.840B.300d.txt"
