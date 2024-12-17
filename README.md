## L3TC
    L3TC: Leveraging RWKV for Learned Lossless Low-Complexity Text Compression

## Requirements

```
pip install -r requirements.txt
```

## Data Preprocess
First, download enwik8 and enwik9 to data/public_text_dataset. Then run the following script to generate dictionary and train/val data.

```
python scripts/preprocessor.py
```

## Train the model

```
python main.py --output_dir work_dirs -c ./config/l3tc/l3tc_200k.py --save_log --amp
```

## Inference && Compression

```
python scripts/compressor.py \
    -c "./config/l3tc/l3tc_200k.py" \
    --pretrain_model_path "work_dirs/l3tc_200k_20241210_135843/checkpoint0019.pth" \
    --tokenizer "dictionary/vocab_enwik8_bpe_16384_0.999/spm_enwik8_bpe_16384_0.999.model" \
    --tmp_processed_dir "data/enwik9_results/l3tc_200k_bpe16k_enwik9" \
    --segment_length 2048 \
    --device cuda \
    --input_file "data/public_text_dataset/enwik9"
```

## Citation
