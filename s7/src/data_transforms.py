from torchvision import transforms


class Transforms:
    def __init__(self, *, normalize=False, mean=None, stdev=None):
        if normalize and not (mean and stdev):
            raise ValueError("Kindly provide mean and standard deviation values, channel wise")
        if not (isinstance(mean, (tuple, list)) and isinstance(stdev, (tuple, list))):
            raise TypeError("Make sure to provide mean and stdev in tuple/list")

        self.normalize = normalize
        self.mean = mean
        self.stdev = stdev

    def __call__(self, *, train=None, pre_transforms=None, post_transforms=None):
        if not (train and isinstance(train, bool)):
            raise ValueError(
                "Please let us know if this transform is for test or train set (use train argument with True/False")

        all_transforms = []
        if self.normalize:
            all_transforms.append(transforms.Normalize(mean=self.mean, std=self.stdev))
        if not train:
            all_transforms = [transforms.ToTensor()] + all_transforms
            return transforms.Compose(all_transforms)

        if pre_transforms:
            all_transforms = [*pre_transforms] + [transforms.ToTensor()]
        if self.normalize:
            all_transforms.append(transforms.Normalize(mean=self.mean, std=self.stdev))
        if post_transforms:
            all_transforms += [*post_transforms]

        return transforms.Compose(all_transforms)
