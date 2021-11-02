import numpy
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.io import Dataset, DataLoader
from inceptionV3 import InceptionV3
import torch.utils.model_zoo as model_zoo
from model import RNN_ENCODER, CNN_ENCODER, G_NET

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=4, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # print(imgs.shape)
    N = len(imgs)
    print(N)
    print(batch_size)
    assert batch_size > 0
    assert N >= batch_size

    # Set up dataloader
    # dataset = MyDataset(imgs)
    # dataloader = DataLoader(dataset, batch_size=1)

    # Load inception model
    model = InceptionV3()
    model.set_state_dict(paddle.load('../../weight/inception_v3.pdparams'))
    up = nn.Upsample(size=(299, 299), mode='bilinear')

    def get_pred(x):
        # print(x.shape)
        if resize:
            x = up(x)
        # print(x.shape)
        print('-'*20)
        print(x.min(), x.max())
        x = model(x)
        # print(len(x[0]))
        # print(x[0].shape)
        print(x[0].min(), x[0].max())
        x = paddle.to_tensor(x[0])
        # print(x.shape)
        return F.softmax(x).detach().cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    # print(len(imgs))
    # for i, batch in enumerate(imgs, 0):
    #     print(i)
    #     print(batch)
    # batch = batch.astype('float32')
    # batch_size_i = batch.shape[0]
    batch = paddle.to_tensor(imgs).astype('float32')
    preds[0: 32] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)

    print("Calculating Inception Score...")
    iscore = inception_score(IgnoreLabelDataset(cifar), cuda=False, batch_size=32, resize=True, splits=10)

