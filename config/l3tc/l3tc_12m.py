### 模型文件路径 ###
source_dir = './data/raw_text_data/'
dataset_name = 'enwik'
vocab_path = './dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999_vocab.json'
train_file = './data/train_data/train_enwik8_bpe_16384_0.999.txt'
test_file = './data/test_data/test_enwik9_bpe_16384_0.999.txt'

### 训练参数 ###
random_seed = 1204
epoch = 20
batch_size = 16
epoch_length = 1000000
checkpoint_path = './checkpoint/'
print_freq = 10
ctx_len = 2048
sentence_length = ctx_len
chunk_size = 1
# scheduler = [None]
# scheduler = ['multi_epoch', [5, 10], 0.1]
scheduler = ['step_lr', 10, 0.9999]
# scheduler = ['exponential_lr', 0.999]
clip_max_norm = 5
save_checkpoint_interval = 1

# optimizer params
betas = (0.9, 0.99)
eps = 1e-8
learning_rate = 1e-4

### 模型参数 ###
model_name = "rwkv_tc_hira"
num_hidden_layer = 4
dropout = 0.2
hidden_size = 384
intermediate_size = 1024
rwkv_rank = 4