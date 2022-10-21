from torch import tensor, int32


class Vocab:
    def __init__(self, frequencies: dict, max_size: int = -1, min_freq: int = 0, special_symbols=None):
        if special_symbols is None:
            special_symbols = {}

        filtered = dict((k, v) for (k, v) in frequencies.items() if v > min_freq and k is not None)
        sliced = sorted(filtered.items(), key=lambda item: item[1], reverse=True)
        if max_size > 0:
            sliced = sliced[:max_size]
        sliced = dict(sliced)
        self.vocabulary = {**special_symbols, **sliced}
        self.stoi = self.transform_stoi()
        self.itos = self.transform_itos()

    def transform_stoi(self) -> dict:
        stoi = {}
        for index, symbol in enumerate(self.vocabulary):
            stoi[symbol] = index
        return stoi

    def transform_itos(self) -> dict:
        itos = {}
        for index, symbol in enumerate(self.vocabulary):
            itos[index] = symbol
        return itos

    def encode(self, symbols) -> tensor:
        indexes = []
        if isinstance(symbols, str):
            return tensor(self.stoi.get(symbols, 1), dtype=int32)

        for s in symbols:
            indexes.append(self.stoi.get(s, 1))
        return tensor(indexes, dtype=int32)

    def decode(self, indexes: list, unk_symbol: str = '<UNK>') -> tensor:
        symbols = []
        for i in indexes:
            symbols.append(self.itos.get(i, unk_symbol))
        return tensor(symbols)
