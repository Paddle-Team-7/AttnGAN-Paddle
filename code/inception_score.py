import numpy
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.io import Dataset, DataLoader
from inceptionV3 import InceptionV3

import numpy as np
from scipy.stats import entropy

import evalDataset


def inception_score(imgdir, batch_size=8, resize=False, splits=10):

    dataset = evalDataset.EvalDataset(imgdir)
    print(imgdir)
    N = len(dataset)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = InceptionV3()
    model.set_state_dict(paddle.load('../models/inception_v3.pdparams'))
    up = nn.Upsample(size=(299, 299), mode='bilinear')

    def get_pred(x):
        if resize:
            x = up(x)
        x = model(x)
        x = paddle.to_tensor(x[0])
        return F.softmax(x).detach().cpu().numpy()

    pred = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        if i % 100 == 0:
            print('{}/{}'.format(i, int(N/batch_size)))
        batch = batch.astype('float32')
        batch_size_i = batch.shape[0]
        pred[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    scores = []
    num_splits = splits
    for index in range(num_splits):
        pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
        kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
        kl_inception = np.mean(np.sum(kl_inception, 1))
        scores.append(np.exp(kl_inception))
    return np.mean(scores), np.std(scores)

    # Now compute the mean kl-div
    # split_scores = []
    # print(len(preds))
    # for k in range(splits):
    #     part = preds[k * (N // splits): (k+1) * (N // splits), :]
    #     py = np.mean(part, axis=0)
    #     scores = []
    #     for i in range(part.shape[0]):
    #         pyx = part[i, :]
    #         scores.append(entropy(pyx, py))
    #     split_scores.append(np.exp(np.mean(scores)))

    # return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    print("Calculating Inception Score...")
    # test_dir = '../models/coco_AttnGAN2_0/valid/single'
    test_dir = '../models/coco_AttnGAN2/valid/single'
    iscore = inception_score(test_dir, batch_size=8, resize=False, splits=10)
    print(iscore)
