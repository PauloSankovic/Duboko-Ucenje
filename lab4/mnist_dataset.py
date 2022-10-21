from collections import defaultdict
from random import choice

import torchvision
from torch.utils.data import Dataset


class MNISTMetricDataset(Dataset):
    def __init__(self, root="./datasets", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            idx = self.targets != remove_class
            self.images = self.images[idx]
            self.targets = self.targets[idx]
            self.classes.remove(remove_class)

        # {class: [img1_index, img2_index...], ...}
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        anchor_class = self.targets[index].item()
        classes = self.classes[:]
        classes.remove(anchor_class)

        c = choice(classes)
        return choice(self.target2indices[c])

    def _sample_positive(self, index):
        anchor_class = self.targets[index].item()

        random_index = index
        while random_index == index:
            random_index = choice(self.target2indices[anchor_class])

        return random_index

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
