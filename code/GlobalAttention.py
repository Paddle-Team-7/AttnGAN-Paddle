"""
Global attention takes a matrix and a query metrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import paddle
import paddle.nn as nn


def conv1x1(in_planes, out_planes):
    """
    1x1 convolution with padding
    """
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=1, padding=0)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.shape[0], query.shape[2]
    ih, iw = context.shape[2], context.shape[3]
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.reshape([batch_size, -1, sourceL])
    contextT = paddle.transpose(context, [0, 2, 1])

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = paddle.bmm(contextT, query) # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.reshape([batch_size*sourceL, queryL])
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.reshape([batch_size, sourceL, queryL])
    # --> batch*queryL x sourceL
    attn = paddle.transpose(attn, [0, 2, 1])
    attn = attn.reshape([batch_size*queryL, sourceL])
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.reshape([batch_size, queryL, sourceL])
    # --> batch x sourceL x queryL
    attnT = paddle.transpose(attn, [0, 2, 1])

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = paddle.bmm(context, attnT)

    return weightedContext, attn.reshape([batch_size, -1, ih, iw])


class GlobalAttentionGeneral(nn.Layer):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.shape[2], input.shape[3]
        queryL = ih * iw
        batch_size, sourceL = context.shape[0], context.shape[2]

        # --> batch x queryL x idf
        target = input.reshape([batch_size, -1, queryL])
        targetT = paddle.transpose(target, [0, 2, 1])
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = paddle.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.reshape([batch_size*queryL, sourceL])
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = paddle.tile(self.mask, [queryL, 1])
            # TODO: data->detach()
            infs = paddle.zeros_like(attn)
            infs += -float('inf')
            attn = paddle.where(mask, infs, attn)
            #attn.detach().masked_fill_(mask.detach(), -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.reshape([batch_size, queryL, sourceL])
        # --> batch x sourceL x queryL
        attn = paddle.transpose(attn, [0, 2, 1])

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = paddle.bmm(sourceT, attn)
        weightedContext = weightedContext.reshape([batch_size, -1, ih, iw])
        attn = attn.reshape([batch_size, -1, ih, iw])

        return weightedContext, attn
