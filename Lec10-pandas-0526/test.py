import torch
import numpy as np

for i in range(1,10,1):
    data = np.load('data_resized_part{}.npy'.format(i))
    # shape: 100000 * 64 * 64
    data = np.expand_dims(data, 3)
    # shape: 10000 * 64 *64 * 1
    data = np.concatenate((data, data, data), 3)
    # shape: 10000 * 64 *64 * 3

    # normalize data
    normalized_data = data/2.

    # save data
    np.save('data_resized_3dim_part{}.npy'.format(i), data)
    np.save('data_resized_3dim_normalized_part{}.npy'.format(i), normalized_data)
    torch.save(torch.from_numpy(normalized_data), 'data_resized_3dim_normalized_part{}.tensor'.format(i))

