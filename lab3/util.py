import numpy as np
from torch import tensor
from torch.nn import Embedding
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from vocab import Vocab


def get_word_frequency_sorted(filepath: str) -> dict:
    data = open(filepath, 'r').read().splitlines()
    word_frequency = {}
    for d in data:
        content, label = d.split(', ')
        words = content.split()
        for word in words:
            if not word:
                continue
            frequency = word_frequency.get(word, 0)
            frequency += 1
            word_frequency[word] = frequency

    return dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True))


def generate_embedding_matrix(vocab: Vocab, filepath: str = None, vec_dim: int = 300, pad_symbol: str = '<PAD>') -> np.ndarray:
    embedding_dict = {}
    if filepath:
        data = open(filepath, 'r').read().splitlines()
        for d in data:
            parts = d.split()
            embedding_dict[parts[0]] = np.array(parts[1:], dtype=np.float32)

    embedding_matrix = np.random.normal(size=(len(vocab.stoi), vec_dim))
    for word, index in vocab.stoi.items():
        if word == pad_symbol:
            embedding_matrix[0] = np.zeros(vec_dim)
        elif word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]

    return embedding_matrix


def matrix_to_torch(matrix: np.ndarray, freeze: bool = True, padding_idx: int = 0) -> Embedding:
    return Embedding.from_pretrained(tensor(matrix), freeze=freeze, padding_idx=padding_idx)


def pad_collate_fn(batch: list, pad_index: int = 0):
    texts, labels = zip(*batch)
    lengths = tensor([len(text) for text in texts])
    texts_pad = pad_sequence(texts, batch_first=True, padding_value=pad_index)  # output = T x B x *
    return texts_pad, tensor(labels), lengths


def get_metrics(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1
