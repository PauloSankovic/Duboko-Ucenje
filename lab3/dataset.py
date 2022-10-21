from torch.utils.data import Dataset

from vocab import Vocab


class NlpDataset(Dataset):
    def __init__(self, filepath: str, data_vocab: Vocab, label_vocab: Vocab):
        data = open(filepath, 'r').read().splitlines()
        instances = []
        for d in data:
            text, label = d.split(", ")
            words = [w for w in text.split() if w]
            instances.append((words, label.strip()))
        self.instances = instances
        self.data_vocab = data_vocab
        self.label_vocab = label_vocab

    def __getitem__(self, item: int):
        words, label = self.instances[item]
        num_words = self.data_vocab.encode(words)
        num_label = self.label_vocab.encode(label)
        return num_words, num_label

    def __len__(self) -> int:
        return len(self.instances)
