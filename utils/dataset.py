# Import Python Modules
import struct
import os.path
import numpy as np


class MNISTLoader:
    def __init__(self, path='.', dataset_name='mnist'):
        self.path = path
        self.dir_path = os.path.join(self.path, 'mnist/')
        self.training_path_dir = os.path.join(self.dir_path, 'train-images-idx3-ubyte')
        self.training_path_target_dir = os.path.join(self.dir_path, 'train-labels-idx1-ubyte')
        self.test_path_dir = os.path.join(self.dir_path, 't10k-images-idx3-ubyte')
        self.test_path_target_dir = os.path.join(self.dir_path, 't10k-labels-idx1-ubyte')

    @staticmethod
    def training_data(data_dir):
        with open(data_dir, 'rb') as data:
            magic, n_samples, n_rows, n_cols = struct.unpack(">IIII", data.read(16))
            data = np.fromfile(data, dtype=np.uint8)
            data = data.reshape(n_samples, n_rows * n_cols)
        data = data.astype(float) / 255.0
        return data

    @staticmethod
    def target_data(target_data_dir):
        with open(target_data_dir, 'rb') as target_data:
            magic, n_samples = struct.unpack(">II", target_data.read(8))
            target = np.fromfile(target_data, dtype=np.int8)
        return target


if __name__ == "__main__":
    loader = MNISTLoader(path='../data/')
    X, X_test = loader.training_data(loader.training_path_dir), loader.training_data(loader.test_path_dir)
    y, y_test = loader.target_data(loader.training_path_target_dir), loader.target_data(loader.test_path_target_dir)
    print(f"Training data shape: {X.shape}, label shape: {y_test.shape}"
          f"test data shape: {X_test.shape}, test label shape: {y_test.shape}")