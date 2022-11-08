# import forestci as fci

from math import log

from fitter import Fitter
from scipy.special import gammaln
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model

from src.model.model import *

my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)
# from src.evaluation.evaluation import *

# optimisers = ['SGD', 'Adam']

rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
warnings.filterwarnings("ignore")

optimisers = ['Adam']
im = 0

precision = []
recall = []
Accuracy = []
F1 = []
force_gc = True


def transform_inverse(data):
    transformer = MinMaxScaler()
    transformer.fit(data)
    # difference transform
    transformed = transformer.transform(data)
    return transformer.inverse_transform(transformed)


def voting(anomalies_rf, anomalies_rnn, anomalies_merged):
    anomalies_rnn1 = anomalies_rnn.values.tolist()
    anomalies_rf1 = anomalies_rf.values.tolist()

    anomalies = set(anomalies_rf1 + anomalies_rnn1)
    anomalies = list(anomalies)
    anomalies = set(anomalies + anomalies_merged)
    # anomalies=pd.DataFrame(anomalies)
    anomalies = list(anomalies)

    append_anomalies = []

    for i in range(len(anomalies)):
        if anomalies[i] in anomalies_rf and anomalies[i] in anomalies_rnn and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])
        elif anomalies[i] in anomalies_rnn and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])
        elif anomalies[i] in anomalies_rf and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])

        elif anomalies[i] in anomalies_rf and anomalies[i] in anomalies_rnn:
            append_anomalies.append(anomalies[i])
    return append_anomalies


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def save_plot_model_rnn(model):
    """
    Saves the plot of the RNN model
    :param model:
    :return:
    """
    plot_model(model, show_shapes=True)


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif dataset == 'SMD':
        return 38
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def save_plot_model_rf(model):
    """
    Saves the plot of the Random Forest model
    :param model:
    :return:
    """
    for i in range(len(model.estimators_)):
        estimator = model.estimators_[i]
        out_file = open("trees/tree-" + str(i) + ".dot", 'w')
        export_graphviz(estimator, out_file=out_file)
        out_file.close()


""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class varU(object):
    """
    EVBUS (Estimate Variance Based on U-Statistics)
    Symbols in the code:
    c, selected c observations, which is the initial fixed points.
    k_n, size of subsamples
    n_MC, number of subsamples | number of trees sharing a observation (L)
    m_n, number of subsamples
    n_z_sim, number of initial sets | number of common observations between trees (B)
    ntree = $L * B$
    @Note: In sklearn, the sub-sample size is always the same as the
    original input sample size but the samples are drawn with
    replacement if bootstrap=True (default).
    """

    def __init__(self, X_train, Y_train, X_test, sub_sample_size=np.nan, n_z_sim=25, n_MC=200, regression=True):
        """
        __init__
        :param X_train: ndarray
        :param Y_train: ndarray
        :param X_test: ndarray
        :param sub_sample_size: int, size of sample to draw, default value is 0.632*n for subsamples
        :param n_z_sim: int, number of common observations between trees (B)
        :param n_MC: int, number of trees sharing a observation (L)
        :param regression: bool, True for regression, False for classification
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.n_z_sim = n_z_sim
        self.n_MC = n_MC
        self.n = X_train.shape[0]
        self.reg = regression
        if sub_sample_size is np.nan:
            self.k_n = int(self.n * 0.3)
        else:
            self.k_n = sub_sample_size

    def sample(self, x, size=None, replace=False, prob=None):
        """
        Take a sample of the specified size from the elements of x using either with or without replacement
        :param x: If an ndarray, a random sample is generated from its elements.
                  If an int, the random sample is generated as if x was np.arange(n)
        :param size: int or tuple of ints, optional. Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
                     ``m * n * k`` samples are drawn. Default is None, in which case a single value is returned.
        :param replace: boolean, optional. Whether the sample is with or without replacement
        :param p: 1-D array-like, optional. The probabilities associated with each entry in x.
                  If not given the sample assumes a uniform distribution over all entries in x.
        :return: 1-D ndarray, shape (size,). The generated random samples
        """
        if not size:
            size = len(x)
        return np.random.choice(x, size, replace, prob)

    def sample_int(self, n, size=0, replace=False, prob=None):
        """
        Return an integer vector of length size with elements from 1:n
        :param n: int, the number of items to choose from
        :param size: int, a non-negative integer giving the number of items to choose
        :param replace: bool, Whether the sample is with or without replacement
        :param prob: 1-D array-like, optional. The probabilities associated with each entry in x.
                     If not given the sample assumes a uniform distribution over all entries in x.
        :return: 1-D ndarray, shape (size,). The generated random samples
        """
        if size == 0:
            size = n
        if replace and prob is None:
            return np.random.randint(1, n + 1, size)
        else:
            return self.sample(range(1, n + 1), size, replace, prob)

    def matrix(self, data=np.nan, nrow=1, ncol=1):
        """
        Build matrix like the operation in R
        :param data:
        :param nrow:
        :param ncol:
        :return:
        """
        if type(data) == int or type(data) == bool:
            if data == 0 or data == False:
                mat = np.zeros((nrow, ncol))
            elif data == 1 or data == True:
                mat = np.ones((nrow, ncol))
            else:
                print("Unsupported int")
                mat = np.zeros((nrow, ncol))
        else:
            mat = np.matrix(data)
        return mat.reshape(nrow, ncol)

    def rep(self, x, times=1, length_out=np.nan, each=1):
        """
        Replicate Elements of Vectors and Lists
        :param x:
        :param times:
        :param length_out:
        :param each:
        :return:
        """
        if each > 1:
            vec = np.repeat(x, each)
        else:
            if type(x) == int or type(x) == float:
                vec = np.repeat(x, times)
            else:
                vec = x * times
        if length_out is not np.nan:
            return vec[:length_out, ]
        else:
            return vec

    def build_subsample_set(self, unique_sample=np.nan, replace=False):
        """
        Build subsample
        :param unique_sample: int, the observation must included in the set
        :param replace: bool, taken with replace or not
        :return: ndarray, indecies of subsamples
        """
        n_train = self.X_train.shape[0]
        if unique_sample is not np.nan:
            sub_sample_candidate = np.delete(np.arange(n_train), unique_sample)
            sub_sample = self.sample(sub_sample_candidate, self.k_n - 1, replace=replace)
            sub_sample = np.append(sub_sample, unique_sample)
        else:
            sub_sample = self.sample(np.arange(n_train), self.k_n, replace=replace)

        return sub_sample

    def estimate_zeta_1_kn(self):
        """
        Estimate $\zeta_{1,k_n}$
        :return: 1d array, variance for the test samples
        """
        n_train = self.X_train.shape[0]
        mean_y_hat = self.matrix(0, self.n_z_sim, self.X_test.shape[0])
        for i in range(self.n_z_sim):
            y_hat = self.matrix(0, self.n_MC, self.X_test.shape[0])

            # Select initial fixed point $\tilde{z}^{(i)}$
            z_i = np.random.randint(0, n_train - 1)

            for j in range(self.n_MC):
                # Select subsample
                # $S_{\tilde{z}^{(i)},j}$ of size $k_n$ from training set that includes $\tilde{z}^{(i)}$
                sub_sample = self.build_subsample_set(z_i)

                x_ss = self.X_train[sub_sample, :]
                y_ss = self.Y_train[sub_sample]

                # Build tree using subsample $S_{\tilde{z}^{(i)},j}$
                if self.reg:
                    tree = RandomForestRegressor(bootstrap=False, n_estimators=1, min_samples_leaf=2)
                else:
                    tree = RandomForestClassifier(bootstrap=False, n_estimators=1, min_samples_leaf=2)
                tree.fit(x_ss, y_ss)

                # Use tree to predict at $x^*$
                y_hat[j, :] = tree.predict(self.X_test)

            # Record average of the $n_{MC}$ predictions
            mean_y_hat[i, :] = np.mean(y_hat, axis=0)

        # Compute the variance of the $n_{\tilde{z}}$ averages
        var_1_kn = np.var(mean_y_hat, axis=0)

        return var_1_kn

    def estimate_zeta_kn_kn(self):
        """
        Estimate $\zeta_{{k_n},{k_n}}$
        :return:
        """
        y_hat = self.matrix(0, self.n_z_sim, self.X_test.shape[0])
        for i in range(self.n_z_sim):
            # select sample of size $k_n$ from training set
            sub_sample = self.build_subsample_set()

            x_ss = self.X_train[sub_sample, :]
            y_ss = self.Y_train[sub_sample]

            # Build tree using subsample $S_{\tilde{z}^{(i)},j}$
            if self.reg:
                tree = RandomForestRegressor(bootstrap=False, n_estimators=1, min_samples_leaf=2)
            else:
                tree = RandomForestClassifier(bootstrap=False, n_estimators=1, min_samples_leaf=2)
            tree.fit(x_ss, y_ss)

            # Use tree to predict at $x^*$
            y_hat[i, :] = tree.predict(self.X_test)

        # Compute the variance of the $n_{\tilde{z}} predictions$
        var_kn_kn = np.var(y_hat, axis=0)
        return var_kn_kn

    def calculate_variance(self, covariance=False):
        """
        Internal Variance Estimation Method
        :param covariance: bool, whether covariance should be returned instead of variance, default is False
        :return: tuple, the first element is the estimation of $\theta_kn$,
                        the second element is the estimation of $variance$ or $covariance$
        """
        n_train = self.X_train.shape[0]
        mean_y_hat = self.matrix(0, self.n_z_sim, self.X_test.shape[0])
        all_y_hat = self.matrix(0, self.n_z_sim * self.n_MC, self.X_test.shape[0])
        for i in range(self.n_z_sim):
            y_hat = self.matrix(0, self.n_MC, self.X_test.shape[0])

            # Select initial fixed point $\tilde{z}^{(i)}$
            z_i = np.random.randint(0, n_train - 1)

            for j in range(self.n_MC):
                # Select subsample
                # $S_{\tilde{z}^{(i)},j}$ of size $k_n$ from training set that includes $\tilde{z}^{(i)}$
                sub_sample = self.build_subsample_set(z_i)

                x_ss = self.X_train[sub_sample, :]
                y_ss = self.Y_train[sub_sample]

                # Build tree using subsample $S_{\tilde{z}^{(i)},j}$
                if self.reg:
                    tree = RandomForestRegressor(bootstrap=False, n_estimators=1, min_samples_leaf=2)
                else:
                    tree = RandomForestClassifier(bootstrap=False, n_estimators=1, min_samples_leaf=2)
                tree.fit(x_ss, y_ss)

                # Use tree to predict at $x^*$
                tmp = tree.predict(self.X_test)
                y_hat[j, :] = tmp
                all_y_hat[i * self.n_MC + j, :] = tmp
            # Record average of the $n_{MC}$ predictions
            mean_y_hat[i, :] = np.mean(y_hat, axis=0)

        if self.reg:
            # Regression
            theta = np.mean(all_y_hat, axis=0)
        else:
            # Classification
            theta = np.zeros(self.X_test.shape[0])

            for i in range(all_y_hat.shape[1]):
                tmp = np.unique(all_y_hat[:, i], return_counts=True)
                max_index = np.argmax(tmp[1])
                theta[i] = tmp[0][max_index]

        # Compute the variance of the $n_{\tilde{z}}$ averages
        m = self.n_MC * self.n_z_sim
        # m = self.n_MC
        alpha = self.n / m
        """
        $variance = \frac{1}{\alpha}\frac{k^2}{m}\zeta_{1,k}+\frac{1}{m}\zeta_{k,k}$
        $\alpha = \frac{n}{m}$
        $variance = \frac{k^2}{n}\zeta_{1,k}+\frac{1}{m}\zeta_{k,k}$
        """
        if covariance:
            cov_1_kn = np.cov(mean_y_hat.T)
            cov_kn_kn = np.cov(all_y_hat.T)
            cov_cp1 = ((self.k_n ** 2) / self.n) * cov_1_kn
            cov_cp2 = cov_kn_kn / m
            cov_u = cov_cp1 + cov_cp2
            return theta, cov_u
        else:
            var_1_kn = np.var(mean_y_hat, axis=0)
            var_kn_kn = np.var(all_y_hat, axis=0)
            var_cp_1 = ((self.k_n ** 2) / self.n) * var_1_kn
            var_cp_2 = var_kn_kn / m
            var_u = var_cp_1 + var_cp_2
            return theta, var_u


def score(REAL_LABELS, PREDICTED_LABELS, var_rf):
    predicter = list(range(len(var_rf)))
    Label_index = REAL_LABELS.loc[REAL_LABELS['0'] > 0]

    a1 = pd.DataFrame(index=range(len(var_rf)), columns=range(2))
    a1.columns = ['index', 'value']

    a2 = pd.DataFrame(index=range(len(var_rf)), columns=range(2))
    a2.columns = ['index', 'value']

    for i in range(len(predicter)):
        if i in PREDICTED_LABELS:
            a1.iloc[i, 1] = 1
        else:

            a1.iloc[i, 1] = 0

    for i in range(len(predicter)):
        if i in Label_index:
            a2.iloc[i, 1] = 1
        else:
            a2.iloc[i, 1] = 0

    y_real = a2.value
    y_real = y_real.astype(int)
    y_predi = a1.value
    y_predi = y_predi.astype(int)
    # pdb.set_trace()
    try:
        cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        precision1 = tp / (tp + fp + + 0.00001)
        recall1 = tp / (tp + fn + + 0.00001)
        Accuracy1 = (tp + tn) / (tp + fn + fp + tn + + 0.00001)
        F11 = 2 / ((1 / precision1) + (1 / recall1))

    except Exception as e:
        print('line 77 evaluation', e)
        precision1 = 0.001
        recall1 = 0.001
        Accuracy1 = 0.001
        F11 = 0.001
    # pdb.set_trace()
    return precision1, recall1, Accuracy1, F11


def Gauss_s(train_score, test_score, train, test, mean_Error_train, mean_error_variance_train, mean_Error_test,
            mean_error_variance_test, logcdf=False):
    f1 = Fitter(train_score, distributions=['norm'])
    f1.fit()
    f1.summary()
    print(f1.fitted_param['norm'])
    f2 = Fitter(test_score, distributions=['norm'])
    f2.fit()
    f2.summary()
    print(f2.fitted_param['norm'])

    params_train_score = f1.fitted_param['norm']
    params_test_score = f2.fitted_param['norm']

    distribution_train = norm(float(params_train_score[0]), float(params_train_score[1]))
    distribution_test = norm(float(params_test_score[0]), float(params_test_score[1]))

    probas_train = distribution_train.logsf(train_score)
    probs1_train = distribution_train.cdf(train_score)
    probs1_test = distribution_test.cdf(test_score)
    probas_test = distribution_test.logsf(test_score)

    train_prob_score = -1 * get_per_channel_probas(train_score.reshape(-1, 1), params_train_score,
                                                   logcdf=logcdf)
    test_prob_score = -1 * get_per_channel_probas(test_score.reshape(-1, 1), params_test_score, logcdf=logcdf)

    return train_prob_score, test_prob_score


def get_scores_channelwise(distr_params, train_raw_scores, val_raw_scores, test_raw_scores, drop_set=set([]),
                           logcdf=False):
    use_ch = list(set(range(test_raw_scores.shape[1])) - drop_set)
    # pdb.set_trace()
    train_prob_scores = -1 * np.concatenate(
        ([get_per_channel_probas(train_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
          for i in range(train_raw_scores.shape[1])]), axis=1)
    test_prob_scores = [get_per_channel_probas(test_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                        for i in range(train_raw_scores.shape[1])]
    test_prob_scores = -1 * np.concatenate(test_prob_scores, axis=1)

    train_ano_scores = np.sum(train_prob_scores[:, use_ch], axis=1)
    test_ano_scores = np.sum(test_prob_scores[:, use_ch], axis=1)
    # pdb.set_trace()

    if val_raw_scores is not None:
        val_prob_scores = -1 * np.concatenate(
            ([get_per_channel_probas(val_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
              for i in range(train_raw_scores.shape[1])]), axis=1)
        val_ano_scores = np.sum(val_prob_scores[:, use_ch], axis=1)
    else:
        val_ano_scores = None
        val_prob_scores = None
    return train_ano_scores, val_ano_scores, test_ano_scores, train_prob_scores, val_prob_scores, test_prob_scores


# Computes (when not already saved) parameters for scoring distributions
def fit_distributions(distr_par_file, distr_names, predictions_dic, val_only=False):
    # try:
    #     with open(distr_par_file, 'rb') as file:
    #         distributions_dic = pickle.load(file)
    # except:
    distributions_dic = {}
    for distr_name in distr_names:
        if distr_name in distributions_dic.keys():
            continue
        else:
            print("The distribution parameters for %s for this algorithm on this data set weren't found. \
            Will fit them" % distr_name)
            if "val_raw_scores" in predictions_dic:
                raw_scores = np.concatenate((predictions_dic["train_raw_scores"], predictions_dic["val_raw_scores"]))
            else:
                raw_scores = predictions_dic["train_raw_scores"]

            distributions_dic[distr_name] = [fit_univar_distr(raw_scores[:, i], distr=distr_name)
                                             for i in range(raw_scores.shape[1])]
    with open(distr_par_file, 'wb') as file:
        pickle.dump(distributions_dic, file)

    return distributions_dic


def fit_univar_distr(scores_arr: np.ndarray, distr='univar_gaussian'):
    """
    :param scores_arr: 1d array of reconstruction errors
    :param distr: the name of the distribution to be fitted to anomaly scores on train data
    :return: params dict with distr name and parameters of distribution
    """
    distr_params = {'distr': distr}
    constant_std = 0.000001
    if distr == "univar_gaussian":
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std == 0.0:
            std += constant_std
        distr_params["mean"] = mean
        distr_params["std"] = std
    elif distr == "univar_lognormal":
        shape, loc, scale = lognorm.fit(scores_arr)
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "univar_lognorm_add1_loc0":
        shape, loc, scale = lognorm.fit(scores_arr + 1.0, floc=0.0)
        if shape == 0.0:
            shape += constant_std
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "chi":
        estimated_df = chi.fit(scores_arr)[0]
        df = round(estimated_df)
        distr_params["df"] = df
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        distr_params["mean"] = mean
        distr_params["std"] = std
    return distr_params


def get_per_channel_probas(pred_scores_arr, params, logcdf=False):
    """
    :param pred_scores_arr: 1d array of the reconstruction errors for one channel
    :param params: must contain key 'distr' and corresponding params
    :return: array of negative log pdf of same length as pred_scores_arr
    """
    dist_params = {}
    dist_params['mean'] = params[0]
    dist_params['std'] = params[1]

    distr = "univar_gaussian"
    probas = None
    constant_std = 0.000001
    # pdb.set_trace()

    if distr == "univar_gaussian":
        assert ("mean" in dist_params.keys() and ("std" in dist_params.keys()) or "variance" in params.keys()), \
            "The mean and/or standard deviation are missing, we can't define the distribution"
        if "std" in dist_params.keys():
            if dist_params["std"] == 0.0:
                dist_params["std"] += constant_std
            distribution = norm(dist_params["mean"], dist_params["std"])

        else:
            distribution = norm(dist_params["mean"], dist_params['std'])
    elif distr == "univar_lognormal":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
    elif distr == "univar_lognorm_add1_loc0":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
        if logcdf:
            probas = distribution.logsf(pred_scores_arr + 1.0)
        else:
            probas = distribution.logpdf(pred_scores_arr + 1.0)
    elif distr == "chi":
        assert "df" in params.keys(), "The number of degrees of freedom is missing, we can't define the distribution"
        df = params["df"]
        distribution = chi(df)
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        assert ("mean" in params.keys() and "std" in params.keys()), "The mean and/or standard deviation are missing, \
        we can't define the distribution"
        distribution = norm(params[0], params[1])

    if probas is None:
        if logcdf:
            probas = distribution.logsf(pred_scores_arr)
        else:
            probas = distribution.logpdf(pred_scores_arr)

    return probas


def eig_method(train_scores, test_scores, pca_var_ratio, var_i):
    ala = []
    bla = []
    # pdb.set_trace()
    th_th = []
    for i in range(len(pca_var_ratio)):
        pcai = pca_var_ratio[i]
        ala.append((train_scores[i].squeeze()) * pcai)
        bla.append((test_scores[i].squeeze()) * pcai)
        varii = np.array(var_i[i])
        # multiplied_list = [element * pcai for element in a_list]
        th_th.append(varii * pcai)
    score_train = np.sum(ala, axis=0) / sum(pca_var_ratio)
    score_test = np.sum(bla, axis=0) / sum(pca_var_ratio)
    # th_train =np.sum(th_th)/sum(pca_var_ratio)
    th_test = np.sum(th_th, axis=0) / sum(pca_var_ratio)

    return score_train, score_test, th_test


def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.

    The returned value will be in [1, n_features - 1].
    """
    ll = np.empty_like(spectrum)
    ll[0] = -np.inf  # we don't want to return n_components = 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return ll.argmax()


def _assess_dimension(spectrum, rank, n_samples):
    """Compute the log-likelihood of a rank ``rank`` dataset.

    The dataset is assumed to be embedded in gaussian noise of shape(n,
    dimf) having spectrum ``spectrum``. This implements the method of
    T. P. Minka.

    Parameters
    ----------
    spectrum : ndarray of shape (n_features,)
        Data spectrum.
    rank : int
        Tested rank value. It should be strictly lower than n_features,
        otherwise the method isn't specified (division by zero in equation
        (31) from the paper).
    n_samples : int
        Number of samples.

    Returns
    -------
    ll : float
        The log-likelihood.

    References
    ----------
    This implements the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
    <https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf>`_
    """

    n_features = spectrum.shape[0]
    if not 1 <= rank < n_features:
        raise ValueError("the tested rank should be in [1, n_features - 1]")

    eps = 1e-15

    if spectrum[rank - 1] < eps:
        # When the tested rank is associated with a small eigenvalue, there's
        # no point in computing the log-likelihood: it's going to be very
        # small and won't be the max anyway. Also, it can lead to numerical
        # issues below when computing pa, in particular in log((spectrum[i] -
        # spectrum[j]) because this will take the log of something very small.
        return -np.inf

    pu = -rank * log(2.)
    for i in range(1, rank + 1):
        pu += (gammaln((n_features - i + 1) / 2.) -
               log(np.pi) * (n_features - i + 1) / 2.)

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2.

    v = max(eps, np.sum(spectrum[rank:]) / (n_features - rank))
    pv = -np.log(v) * n_samples * (n_features - rank) / 2.

    m = n_features * rank - rank * (rank + 1.) / 2.
    pp = log(2. * np.pi) * (m + rank) / 2.

    pa = 0.
    spectrum_ = spectrum.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(spectrum)):
            pa += log((spectrum[i] - spectrum[j]) *
                      (1. / spectrum_[j] - 1. / spectrum_[i])) + log(n_samples)

    ll = pu + pl + pv + pp - pa / 2. - rank * log(n_samples) / 2.

    return ll
