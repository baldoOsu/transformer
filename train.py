import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizesr.trainers import WordLeveLTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds):
    for item in ds:
        yield item['text']

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'].format(config['dataset_name']))
    if not Path.exists(tokenizer_path):
        # unk_token is used when a word is not in the vocabulary
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pro_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    ds_raw = load_dataset()