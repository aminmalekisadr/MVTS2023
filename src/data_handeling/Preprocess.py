import gc  # garbage collector
import pdb

import numpy as np
import pandas as pd
# import forestci as fci
import tensorflow
from keras.models import Model
from scipy import linalg
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap, MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import svd_flip, stable_cumsum

from src.utils import _infer_dimension
from src.utils import autoencodermodel

my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
#
import yaml
import numbers


class Preprocess:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, "r") as yaml_config_file:
            self.experiment_config = yaml.safe_load(yaml_config_file)
        self.train = None
        self.test = None

    def collect_gc(self):
        """
        Forces garbage collector
        :return:
        """
        if self.experiment_config['ga_parameters']['force_gc']:
            gc.collect()

    def build_df(self, data, start=0, name='SMAP'):
        #    pdb.set_trace()
        index = np.array(range(start, start + len(data)))
        timestamp = index * 86400 + 1022819200
        if name == 'SMD':
            data = pd.DataFrame(data)
            data['timestamp'] = timestamp.astype(int)
            data['index'] = index.astype(int)
            return data
        return pd.DataFrame({'timestamp': timestamp.astype(int), 'value': data[:, 0], 'index': index.astype(int)})

    def create_sliding_window(self, data, sequence_length, stride=1):
        X_list, y_list = [], []
        for i in range(len(data)):
            if (i + sequence_length) < len(data):
                X_list.append(data.iloc[i:i + sequence_length:stride, :].values)
                y_list.append(data.iloc[i + sequence_length, -1])
        return np.array(X_list), np.array(y_list)

    def inverse_transform(self, y):
        return self.scaler.inverse_transform(y.reshape(-1, 1))

    def create_dataset(self, dataset, look_back):
        """
        Converts an array of values into a dataset matrix
        :param dataset:
        :param look_back:
        :return:
        """
        dataX, dataY = [], []
        # pdb.set_trace()
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[(i + look_back):(i + look_back + 1)])
        self.collect_gc()

        return np.array(dataX), np.array(dataY)

    def preprocess_algorithm(self, data):

        sequence_length = self.experiment_config['data_parameters']['look_back3']

        gama = 1
        m = 2
        t = 0
        X = []
        G = []
        while t in range(len(data)):
            data_t = data[t:t + sequence_length]
            # pdb.set_trace()
            data_t = data_t.value

            mean = np.mean(data[t:t + sequence_length].value)
            std = np.std(data[t:t + sequence_length].value)
            threshold = mean + 2 * std

            I = np.arange(start=0, stop=m)
            I = list(I)
            # pdb.set_trace()
            for j in I:
                # print(j)
                if j == 0:
                    et = data_t[j]
                else:
                    et = np.abs(data_t[j] - data_t[j - 1])

                    # print(j)
                    # pdb.set_trace()
                if et <= threshold:
                    gama = np.exp(-et)
                    m = m + 1
                    X.append(gama * data_t[j])
                    G.append(gama)
                else:
                    break
        # pdb.set_trace()

        Processed_data = np.sum(X) / norm(G)
        t = t + 1
        return Processed_data

    def load_dataset(self, train):
        """
        Loads a dataset with training and testing arrays
        :param dataset_path:
        :return:
        """
        # Load dataset
        # pdb.set_trace()
        # if train_flag:
        self.dataset = train
        #
        # self.dataset = self.dataset.drop(columns=['timestamp', 'index', 'name'])
        # pdb.set_trace()

        # self.dataset = np.array(self.dataset.values)

        # dataset = dataset.value  # as numpy array
        self.dataset = self.dataset.astype('float64')
        # pdb.set_trace()

        # Normalise the dataset
        scaler = MinMaxScaler(feature_range=(-1, 1))

        self.dataset = scaler.fit_transform(self.dataset)

        # pdb.set_trace()

        # split into train and test sets
        # train_size = int(len(self.dataset) * self.experiment_config['data_parameters']['train_size_percentage'])

        # train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]

        # reshape into X=t and Y=t+1
        # pdb.set_trace()

        train_x, train_y = self.create_dataset(train, self.experiment_config['data_parameters']['look_back'])
        # test_x, test_y = self.create_dataset(test, self.experiment_config['data_parameters']['look_back'])
        # reshape input to be [samples, time steps, features]
        train_x_rf, train_y_rf = self.create_dataset(train, self.experiment_config['data_parameters']['look_back2'])
        # test_x_rf, test_y_rf = self.create_dataset(test, self.experiment_config['data_parameters']['look_back2'])
        # pdb.set_trace()
        train_x_rf_stf = train_x_rf  # .reshape(train_x_rf.shape[0], train_x_rf.shape[1], 1)
        # test_x_rf_stf = test_x_rf  # .reshape(test_x_rf.shape[0], test_x_rf.shape[1], 1)
        train_x_lr, train_y_lr = self.create_dataset(train, self.experiment_config['data_parameters']['look_back3'])
        # test_x_lr, test_y_lr = self.create_dataset(test, self.experiment_config['data_parameters']['look_back3'])
        # pdb.set_trace()

        train_x_stf = train_x
        # np.reshape(train_x, (train_x.shape[0], train_x.shape[2], train_x.shape[1]))
        # test_x_stf = test_x
        # np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        train_x_st = train_x  # np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
        # test_x_st = test_x  # np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        # pdb.set_trace()
        train_x_rf_st = train_x_rf  # np.reshape(train_x_rf, (train_x_rf.shape[0], train_x_rf.shape[1]))
        # test_x_rf_st = test_x_rf  # np.reshape(test_x_rf, (test_x_rf.shape[0], test_x_rf.shape[1]))
        train_x_lr_st = train_x_lr  # np.reshape(train_x_lr, (train_x_lr.shape[0], train_x_lr.shape[1]))
        # test_x_lr_st = test_x_lr  # np.reshape(test_x_lr, (test_x_lr.shape[0], test_x_lr.shape[1]))

        return scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr

    def pca(self, train, test, data_path, n_components=0.95, show_plots=False):
        """
        Performs PCA on the data
        :param data:
        :param n_components:
        :return:
        """
        pca_train = PCA(n_components=0.9)
        pca_train.fit(train)

        self.train = pca_train.transform(train)

        # self.train = pca_train.transform(train)

        pca_test = PCA(n_components=pca_train.n_components_)
        pca_test.fit(test)

        self.test = pca_test.transform(test)
        pdb.set_trace()

        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)

        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        if show_plots:
            import matplotlib.pyplot as plt
            plt.plot(pca_train.explained_variance_ratio_)
            plt.xlabel('Number of components')
            plt.ylabel('Variance (%)')

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, pca_test.explained_variance_, pca_test.explained_variance_ratio_

    def svd(self, train, test, data_path, n_components=0.95, show_plots=False):
        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')
        svd_train = TruncatedSVD(n_components=train_n_components)
        svd_train.fit(train)

        self.train = svd_train.transform(train)

        # self.train = pca_train.transform(train)

        svd_test = TruncatedSVD(n_components=test_n_components)
        svd_test.fit(test)
        self.test = svd_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    def manifold(self, train, test, data_path, n_components=0.95, show_plots=False):
        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')
        manifold_train = LocallyLinearEmbedding(n_components=train_n_components)
        manifold_train.fit(train)

        self.train = manifold_train.transform(train)

        manifold_test = LocallyLinearEmbedding(n_components=test_n_components)
        manifold_test.fit(test)
        self.test = manifold_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    def isomap(self, train, test, data_path, n_components=0.95, show_plots=False):
        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')

        Isomap_train = Isomap(n_components=train_n_components)

        Isomap_train.fit(train)

        self.train = Isomap_train.transform(train)

        Isomap_test = Isomap(n_components=test_n_components)
        Isomap_test.fit(test)
        self.test = Isomap_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    def tsne(self, train, test, data_path, n_components=0.95, show_plots=False):
        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')
        tsne_train = TSNE(n_components=train_n_components, learning_rate=300, perplexity=30, early_exaggeration=12,
                          init='random',
                          random_state=2019, n_iter=1000, n_iter_without_progress=20, metric='euclidean', verbose=0,
                          method='exact')
        self.train = tsne_train.fit_transform(train)

        tsne_test = TSNE(n_components=test_n_components, learning_rate=300, perplexity=30, early_exaggeration=12,
                         init='random',
                         random_state=2019, n_iter=1000, n_iter_without_progress=20, metric='euclidean', verbose=0,
                         method='exact')
        self.test = tsne_test.fit_transform(test)

        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    def spectralembeding(self, train, test, data_path, n_components=0.95, show_plots=False):
        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')

        spectral_train = SpectralEmbedding(n_components=train_n_components)
        spectral_train.fit(train)

        self.train = spectral_train.fit_transform(train)

        spectral_test = SpectralEmbedding(n_components=test_n_components)
        spectral_test.fit(test)
        self.test = spectral_test.fit_transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    def ipca(self, train, test, data_path, n_components=0.95, show_plots=False):
        ipca_train = IncrementalPCA(n_components=0.9)
        ipca_train.fit(train)
        self.train = ipca_train.transform(train)
        ipca_test = IncrementalPCA(n_components=self.train.n_components_)
        ipca_test.fit(test)
        self.test = ipca_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, ipca_test.explained_variance_, ipca_test.explained_variance_ratio_

    def kernel_pca(self, train, test, data_path, n_components=0.95, show_plots=False):
        kpca_train = KernelPCA(n_components=0.9)
        kpca_train.fit(train)
        self.train = kpca_train.transform(train)
        kpca_test = KernelPCA(n_components=self.train.n_components_)
        kpca_test.fit(test)
        self.test = kpca_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, kpca_test.explained_variance_, kpca_test.explained_variance_ratio_

    def sparse_pca(self, train, test, data_path, n_components=0.95, show_plots=False):
        spca_train = SparsePCA(n_components=0.9)
        spca_train.fit(train)
        self.train = spca_train.transform(train)
        spca_test = SparsePCA(n_components=self.train.n_components_)
        spca_test.fit(test)
        self.test = spca_test.transform(test)
        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, spca_test.explained_variance_, spca_test.explained_variance_ratio_

    def mds(self, train, test, data_path, n_components=0.95, show_plots=False):

        train_explained_variance_, train_explained_variance_ratio_, train_n_components = self.explain_var(train,
                                                                                                          n_components='mle')
        test_explained_variance_, test_explained_variance_ratio_, test_n_components = self.explain_var(test,
                                                                                                       n_components='mle')

        mds_train = MDS(n_components=train_n_components, n_init=12, max_iter=700, metric=True, n_jobs=4,
                        random_state=2019)
        mds_train.fit(train)
        self.train = mds_train.fit_transform(train)

        mds_test = MDS(n_components=test_n_components, n_init=12, max_iter=700, metric=True, n_jobs=4,
                       random_state=2019)
        mds_test.fit(test)
        self.test = mds_test.fit_transform(test)

        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_, test_explained_variance_, test_explained_variance_ratio_

    def auto_encoder(self, model, train, test, data_path, show_plots=False):

        # m.add(Dense(500, activation='elu', input_shape=(38, 200)))

        # input_shape = train_x_st.shape[2]
        model1 = autoencodermodel(input_shape=train.shape[1])
        history1 = model1.fit(train, train, epochs=1, verbose=0, batch_size=1)
        encoder1 = Model(model1.input, model1.get_layer('encoder_3').output)
        self.train = encoder1.predict(train)
        Renc1 = model1.predict(train)

        model2 = autoencodermodel(input_shape=test.shape[1])
        history2 = model2.fit(test, test, epochs=1, verbose=0, batch_size=1)
        encoder2 = Model(model2.input, model2.get_layer('encoder_3').output)
        self.test = encoder2.predict(test)
        Renc2 = model2.predict(test)

        train_explained_variance_, train_explained_variance_ratio_ = self.explain_var(self.train, n_components=5)
        test_explained_variance_, test_explained_variance_ratio_ = self.explain_var(self.test, n_components=5)

        # pdb.set_trace()

        scaler, train_x_stf, train_x_st, train_y, train_x_rf_stf, train_x_rf, train_y_rf, train_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr = self.load_dataset(
            self.train)
        scaler, test_x_stf, test_x_st, test_y, test_x_rf_stf, test_x_rf, test_y_rf, test_x_rf_st, test_x_lr_st, test_x_lr, test_y_lr = self.load_dataset(
            self.test)
        train_x, train_y = self.create_dataset(self.train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(self.test, self.experiment_config['data_parameters']['look_back'])

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st, train_x_lr_st, train_x_lr, train_y_lr, test_x_lr_st, test_x_lr, test_y_lr, test_explained_variance_, test_explained_variance_ratio_

    @staticmethod
    def explain_var(X, n_components):

        n_samples, n_features = X.shape

        if n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

        mean_ = np.mean(X, axis=0)
        X -= mean_

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.
        # Postprocess the number of components required
        if n_components == 'mle':
            n_components = \
                _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            noise_variance_ = explained_variance_[n_components:].mean()
        else:
            noise_variance_ = 0
            n_samples_, n_features_ = n_samples, n_features

        components_ = components_[:n_components]
        n_components_ = n_components
        explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        singular_values_ = singular_values_[:n_components]
        return explained_variance_, explained_variance_ratio_, n_components_
