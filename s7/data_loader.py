import torch


class EVADataLoader:
    def __init__(self, batch_size=256, shuffle=True, seed=101, num_workers=2):
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.manual_seed(seed)
        self.data_loader_args = dict(shuffle=shuffle, batch_size=batch_size,
                                     num_workers=num_workers, pin_memory=True) if cuda else \
            dict(shuffle=shuffle,
                 batch_size=batch_size)

    def __call__(self, *, data):
        return torch.utils.data.DataLoader(data, **self.data_loader_args)
