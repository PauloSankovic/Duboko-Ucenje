import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NlpDataset
from util import get_word_frequency_sorted, generate_embedding_matrix, pad_collate_fn, matrix_to_torch
from vocab import Vocab

special_symbols = {'<PAD>': 0, '<UNK>': 1}
label_frequency = {"positive": 2, "negative": 1}


def vocab_data_loader(seed, vocab_max_size, vocab_min_freq, train_bs, valid_bs, test_bs, shuffle):
    np.random.seed(seed)
    torch.manual_seed(seed)

    frequencies = get_word_frequency_sorted("./data/sst_train_raw.csv")
    data_vocab = Vocab(frequencies, max_size=vocab_max_size, min_freq=vocab_min_freq, special_symbols=special_symbols)
    label_vocab = Vocab({}, max_size=vocab_max_size, min_freq=vocab_min_freq, special_symbols=label_frequency)

    train_dataset = NlpDataset('./data/sst_train_raw.csv', data_vocab=data_vocab, label_vocab=label_vocab)
    valid_dataset = NlpDataset('./data/sst_valid_raw.csv', data_vocab=data_vocab, label_vocab=label_vocab)
    test_dataset = NlpDataset('./data/sst_test_raw.csv', data_vocab=data_vocab, label_vocab=label_vocab)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=shuffle,
                                  collate_fn=pad_collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_bs, shuffle=shuffle,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=shuffle, collate_fn=pad_collate_fn)

    embeddings = generate_embedding_matrix(vocab=data_vocab, filepath='./data/sst_glove_6b_300d.txt')
    embeddings = matrix_to_torch(embeddings)
    return train_dataloader, valid_dataloader, test_dataloader, embeddings
