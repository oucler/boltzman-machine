# Import Python Modules
import struct
import os.path
import numpy as np


class MNISTLoader:
    def __init__(self, path='.'):
        self.path = path

    def load(self, mode='train'):
        """
        Load and return MNIST dataset.

        Returns
        -------
        data : (n_samples, 784) np.ndarray
            Data representing raw pixel intensities (in [0., 255.] range).
        target : (n_samples,) np.ndarray
            Labels vector (zero-based integers).
        """
        dirpath = os.path.join(self.path, 'mnist/')
        if mode == 'train':
            fname_data = os.path.join(dirpath, 'train-images-idx3-ubyte')
            fname_target = os.path.join(dirpath, 'train-labels-idx1-ubyte')
        elif mode == 'test':
            fname_data = os.path.join(dirpath, 't10k-images-idx3-ubyte')
            fname_target = os.path.join(dirpath, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError("`mode` must be 'train' or 'test'")

        with open(fname_data, 'rb') as fdata:
            magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", fdata.read(16))
            data = np.fromfile(fdata, dtype=np.uint8)
            data = data.reshape(n_samples, n_rows * n_cols)

        with open(fname_target, 'rb') as ftarget:
            magic, n_samples = struct.unpack(">II", ftarget.read(8))
            target = np.fromfile(ftarget, dtype=np.int8)

        data = data.astype(float) / 255.0
        return data, target


if __name__ == "__main__":
    loader = MNISTLoader(path='../data/')
    X, y = loader.load(mode='train')
    X_test, y_test = loader.load(mode='test')
    print(X.shape, y.shape, X_test.shape, y_test.shape)