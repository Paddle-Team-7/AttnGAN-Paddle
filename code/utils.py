from collections import namedtuple
import paddle

PackedSequence_ = namedtuple('PackedSequence', ['data', 'batch_sizes'])


class PackedSequence(PackedSequence_):

    pass


def pack_padded_sequence(input, lengths, batch_first=False):
    if lengths[-1] <= 0:
        raise ValueError("length of all samples has to be greater than 0, "
                         "but found an element in 'lengths' that is <=0")
    if batch_first:
        # print(input.shape)
        input = input.transpose([1, 0, 2])

    steps = []
    batch_sizes = []
    lengths_iter = reversed(lengths)
    current_length = next(lengths_iter)
    batch_size = input.shape[1]
    if len(lengths) != batch_size:
        raise ValueError("lengths array has incorrect size")

    for step, step_value in enumerate(input, 1):
        steps.append(step_value[:batch_size])
        batch_sizes.append(batch_size)

        while step == current_length:
            try:
                new_length = next(lengths_iter)
            except StopIteration:
                current_length = None
                break

            if current_length > new_length:  # remember that new_length is the preceding length in the array
                raise ValueError("lengths array has to be sorted in decreasing order")
            batch_size -= 1
            current_length = new_length
        if current_length is None:
            break
    return PackedSequence(paddle.concat(steps), batch_sizes)


def pad_packed_sequence(sequence, batch_first=False):
    var_data, batch_sizes = sequence
    max_batch_size = batch_sizes[0]
    output = var_data.data.new(len(batch_sizes), max_batch_size, *var_data.shape[1:]).zero_()

    lengths = []
    data_offset = 0
    prev_batch_size = batch_sizes[0]
    for i, batch_size in enumerate(batch_sizes):
        output[i, :batch_size] = var_data[data_offset:data_offset + batch_size]
        data_offset += batch_size

        dec = prev_batch_size - batch_size
        if dec > 0:
            lengths.extend((i,) * dec)
        prev_batch_size = batch_size
    lengths.extend((i + 1,) * batch_size)
    lengths.reverse()

    if batch_first:
        output = output.transpose([1, 0, 2])
    return output, lengths
