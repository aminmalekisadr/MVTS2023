import pdb
import random
import sys
import time

import numpy as np
import pandas as pd
# import forestci as fci
import tensorflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.utils import Gauss_s
from src.utils import eig_method
from src.utils import score

my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)
from mlinsights.mlmodel import IntervalRegressor
import matplotlib.pyplot as plt
import logging
import yaml

with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
# Load the data


rnn_epochs = experiment_config['ml_parameters']['rnn_epochs']
optimiser = experiment_config['ml_parameters']['optimizer_type']
mcdropout_samples = experiment_config['ml_parameters']['mc_dropout']
import plotly.graph_objects as go
from tqdm import tqdm


def figure(test_uncertainty_df, err_up, err_down, test_predict, test_y):
    test_uncertainty_df['value_mean'] = test_uncertainty_df.filter(like='value', axis=1).mean(axis=1)
    test_uncertainty_df['value_std'] = test_uncertainty_df.filter(like='value', axis=1).std(axis=1)

    test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - 3 * abs(
        test_uncertainty_df['value_std'])
    test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + 3 * abs(
        test_uncertainty_df['value_std'])

    test_uncertainty_df['lower_bound'] = err_down
    test_uncertainty_df['upper_bound'] = err_up
    # pdb.set_trace()
    # evbus = varU(train_x, train_y, test_x)

    # v = evbus.calculate_variance()

    test_uncertainty_df['index'] = pd.DataFrame(test_predict).index

    test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = pd.DataFrame()

    truth_uncertainty_plot_df['value'] = test_y
    truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

    # .copy(deep=True)
    # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

    upper_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['upper_bound'],
        mode='lines',
        fill=None,
        name='99% Upper Confidence Bound 3xSTD',
    )
    lower_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['lower_bound'],
        mode='lines',
        fill='tonexty',
        name='99% Lower Confidence Bound 3xSTD',
        fillcolor='rgba(255, 211, 0, 0.1)',
    )
    real_trace = go.Scatter(
        x=truth_uncertainty_plot_df['index'],
        y=truth_uncertainty_plot_df['value'],
        mode='lines',
        fill=None,
        name='Real Values'
    )

    data = [upper_trace, lower_trace, real_trace]

    fig = go.Figure(data=data)
    fig.update_layout(title='Uncertainty',
                      xaxis_title='index',
                      yaxis_title='value',
                      legend_font_size=14,
                      # paper_bgcolor='rgba(0,0,0,0)',
                      # plot_bgcolor ='rgba(0,0,0,0)',
                      )
    # fig.show()


def pred_ints(model, X, percentile=99):
    err_down = []
    err_up = []
    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x].reshape(1, -1))[0])

        err_down.append(np.percentile(preds, (100 - percentile) / 2.))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)


def predict_dist(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    # pdb.set_trace()
    return np.hstack(preds)


def predict_point(X, model, num_samples):
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist.mean(axis=1), pred_dist.var(axis=1)


def evaluate_lr(model, train_x, test_x, train_y, test_y, scaler, label_data, pca_variance, pca_var_ratio) -> tuple:
    """
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :label_data:
    :return:
    """

    min_estimators = experiment_config['ml_parameters']['min_number_of_estimators_LR']
    max_estimators = experiment_config['ml_parameters']['max_number_of_estimators_LR']
    alpha_LR_min = experiment_config['ml_parameters']['alpha_LR_min']
    alpha_LR_max = experiment_config['ml_parameters']['alpha_LR_max']
    alpha_LR = random.uniform(alpha_LR_min, alpha_LR_max)
    num_estimators = random.randint(min_estimators, max_estimators)
    # Evaluate the model
    test_uncertainty_df = pd.DataFrame()

    # LR

    # train_x = [train_x[i][0] for i in range(train_x.shape[0])]
    # train_x=  np.array(train_x).reshape(-1,1)
    train_x = np.array(train_x)
    # test_x = [test_x[i][0] for i in range(test_x.shape[0])]
    test_x = np.array(test_x)  # .reshape(-1,1)
    train_y = np.array(train_y)  # .reshape(-1, 1)
    test_y = np.array(test_y)  # .reshape(-1, 1)
    #    pdb.set_trace()
    # nsamples, nx, ny = train_x.shape

    # train_x = train_x.reshape((nsamples, nx * ny))
    # nsamples1, nx1, ny1 = test_x.shape

    # test_x = test_x.reshape((nsamples1, nx1 * ny1))

    # lr_types = ['LinearRegression', 'LinearRegression_Higherconfidence_region', 'DecisionTreeRegressor']
    # lr_type_index = random.randint(0, len(lr_types) - 1)
    # lr_type = lr_types[lr_type_index]
    lr_type = 'LinearRegression'
    vari = []
    train_prob_scores_total = []
    test_prob_scores_total = []
    test_uncertainty_mean = []
    test_uncertainty_std = []
    train_scores_total = []
    test_scores_total = []
    truth_uncertainty_df = pd.DataFrame()
    train_predict = []
    test_predict = []

    for k in range(train_x.shape[2]):
        print(k)
        time_start = time.time()
        # pdb.set_trace()
        model = IntervalRegressor(LinearRegression(normalize=True), n_estimators=num_estimators, alpha=alpha_LR)
        train_x1 = train_x[:, :, k]  # .reshape((nsamples, nx))

        # nsampless, nx = train_y[:, k].shape

        train_y1 = train_y[:, :, k]  # .reshape((nsampless, nx))

        # nsamples1, nx1 = test_x[:, k, :].shape

        test_x1 = test_x[:, :, k]  # .reshape((nsamples1, nx1))
        # nsamples1, nx1 = test_y[:, k].shape

        test_y1 = test_y[:, :, k]  # .reshape((nsamples1, nx1))
        # pdb.set_trace()
        model.fit(train_x1, train_y1.squeeze())
        time_end = time.time()
        train_time = time_end - time_start
        time_start = time.time()
        pred_y = model.predict(test_x1)
        time_end = time.time()
        test_time = time_end - time_start
        # train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
        train_y1 = scaler.inverse_transform(train_y1.reshape(-1, 1))
        test_predict1 = scaler.inverse_transform(pred_y.reshape(-1, 1))
        test_y1 = scaler.inverse_transform(test_y1.reshape(-1, 1))
        # pdb.set_trace()

        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        # test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x1)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
        # sns.displot(data=model.predict(test_x), kind="hist", bins=100, aspect=1.5)
        # plt.show()
        # pdb.set_trace()
        variance_lr = abs(max_pred - min_pred) / 6.0
        # pdb.set_trace()

        train_predict1 = model.predict(train_x1)  # mpg x_hat)
        test_predict1 = model.predict(test_x1)
        # mpg y_hat)
        # train_predict = trans.transform(train_x)  # .reshape(-1,1) #mpg x_hat)
        # test_predict = pred_combine

        train_predict1 = scaler.inverse_transform([train_predict1])
        train_y1 = scaler.inverse_transform(train_y1)
        # test_predict = scaler.inverse_transform([test_predict])
        # test_y = scaler.inverse_transform([test_y])
        # Calculate RMSE for train and test

        test_uncertainty_df['index'] = test_uncertainty_df.index
        # import plotly.graph_objects as go

        test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
        # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
        truth_uncertainty_plot_df = pd.DataFrame()

        truth_uncertainty_plot_df['value'] = test_y1[0]
        truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

        test_uncertainty_df['value_mean'] = test_predict1
        # test_uncertainty_df['value_std'] = result.bse.mean(axis=0)
        err_down, err_up = pred_ints(model, test_x1)

        test_uncertainty_df = test_uncertainty_df[['value_mean']]
        test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - variance_lr
        test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + variance_lr
        test_uncertainty_df['lower_bound'] = err_up
        test_uncertainty_df['upper_bound'] = err_down
        # pdb.set_trace()
        truth_uncertainty_df['value'] = test_y1[:, 0]
        truth_uncertainty_df['index'] = truth_uncertainty_df.index
        # pdb.set_trace()
        # test_predict = scaler.inverse_transform(test_predict)

        #  test_y = scaler.inverse_transform([test_y])
        # Calculate RMSE for train and test
        #    train_score = mean_squared_error(train_y[:, 0], train_predict[:, 0])
        train_scores = np.sqrt((train_y1.squeeze() - train_predict1.squeeze()) ** 2)
        test_scores = np.sqrt((test_y1.squeeze() - test_predict1.squeeze()) ** 2)
        #     test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
        # pdb.set_trace()

        # train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
        # test_score_MC = mean_squared_error(test_y[0], test_MC_predict_point[0, :])
        # print('Train Score: %.2f RMSE' % (train_score))
        #   test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
        # print('Test Score: %.2f RMSE' % (test_score))
        # pdb.set_trace()
        train_score_mean = np.mean(train_scores)
        train_score_std = np.std(train_scores)
        test_score_mean = np.mean(test_scores)
        test_score_std = np.std(test_scores)
        variance_lr = (np.array(max_pred) - np.array(min_pred)) / 6.0

        # pdb.set_trace()

        # pdb.set_trace()
        train_predict1 = train_predict1.reshape(-1, 1)
        test_predict1 = test_predict1.reshape(-1, 1)
        train_prob_scores, test_prob_scores = Gauss_s(train_scores, test_scores, train_predict, test_predict1,
                                                      train_score_mean, train_score_std,
                                                      test_score_mean, test_score_std)

        # pdb.set_trace()
        vari.append(variance_lr)
        train_prob_scores_total.append(train_prob_scores)
        test_prob_scores_total.append(test_prob_scores)
        test_uncertainty_mean.append(test_uncertainty_df['value_mean'])
        # test_uncertainty_std.append(test_uncertainty_df['value_std'])
        train_scores_total.append(train_scores)
        test_scores_total.append(test_scores)
        train_ano_scores = np.sum(train_prob_scores_total, axis=0)
        test_ano_scores = np.sum(test_prob_scores_total, axis=0)
        test_predict.append(test_predict1)
        train_predict.append(train_predict1)

    # fig.show()
    _, anomaly_score_eig, th_eig = eig_method(train_prob_scores_total, test_prob_scores_total, pca_var_ratio, vari)

    bounds_df = pd.DataFrame()
    test_uncertainty_df['lower_bound'] = -  th_eig

    test_uncertainty_df['upper_bound'] = th_eig
    bounds_df = pd.DataFrame()

    bounds_df['lower_bound'] = test_uncertainty_df['lower_bound']
    # bounds_df['prediction'] = test_uncertainty_df['value_mean']
    # bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df['upper_bound'] = test_uncertainty_df['upper_bound']
    # pdb.set_trace()

    bounds_df['contained'] = (bounds_df['upper_bound'] >= anomaly_score_eig.squeeze())

    print("Proportion of points contained within 99% confidence interval:",
          bounds_df['contained'].mean())
    # pdb.set_trace()
    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    # pdb.set_trace()

    model.train_score = train_scores_total
    model.test_score = test_scores_total
    # model.train_score_MC = train_score_MC
    # model.test_score_MC = test_score_MC
    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()
    training_truth_df = pd.DataFrame()
    testing_truth_df = pd.DataFrame()

    #        training_df['value'] = train_MC_predict_point[0]
    #        training_df['index'] = training_df.index
    #       training_df['source'] = 'Training Prediction'
    #       testing_df['value'] = test_MC_predict_point[0]
    #      testing_df['index'] = testing_df.index
    #     testing_df['source'] = 'Test Prediction'
    #    training_truth_df['value'] = train_y[0]
    #    training_truth_df['index'] = training_truth_df.index
    #    training_truth_df['source'] = 'True Value Training'
    #    testing_truth_df['value'] = test_y[0]
    #    testing_truth_df['index'] = testing_truth_df.index
    #    testing_truth_df['source'] = 'True Value Testing'
    # pdb.set_trace()

    #   evaluation = pd.concat([training_df,
    #                          testing_df,
    #                         training_truth_df,
    #                        testing_truth_df
    #                       ], axis=0)

    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    # N = Nmax_gen(predictedanomaly, test_uncertainty_df, truth_uncertainty_plot_df, label_data, name)

    # Using 99% confidence bounds
    #    bounds_df['lower_bound'] = test_uncertainty_df['lower_bound']
    #   bounds_df['prediction'] = test_uncertainty_df['value_mean']
    #  bounds_df['real_value'] = test_y
    # bounds_df['upper_bound'] = test_uncertainty_df['upper_bound']

    # bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
    #                         (bounds_df['real_value'] <= bounds_df['upper_bound']))

    # print("Proportion of points contained within 99% confidence interval:",
    #     bounds_df['contained'].mean())
    # predictedanomaly = bounds_df.index[~bounds_df['contained']]
    print("Number of points predicted to be anomalies:", len(predictedanomaly))
    #  pdb.set_trace()
    # plt.style.use("seaborn-whitegrid")
    # plt.rc(
    #   "figure",
    #  autolayout=True,
    # figsize=(11, 4),
    # titlesize=18,
    # titleweight='bold',
    # )
    # plt.rc(
    #   "axes",
    #  labelweight="bold",
    # labelsize="large",
    # titleweight="bold",
    # titlesize=16,
    # titlepad=10,
    # )
    # fig, ax = plt.subplots()
    #    ax.plot('index', 'values', data=test_uncertainty_df, color='0.75')
    # ax = sns.regplot(x='index', y='values', data=test_uncertainty_df, ci=None, scatter_kws=dict(color='0.25'))
    # pdb.set_trace()

    #    test_predict = scaler.inverse_transform(test_predict)
    #  test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    # pdb.set_trace()

    # train_score = mean_squared_error(train_y[0], train_predict[0])
    # train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
    # train_scorevar = train_MC_predict_point[1, :]
    # test_scorevar = test_MC_predict_point[1, :]
    # test_score_MC = mean_squared_error(test_y, test_MC_predict_point[0, :])
    # print('Train Score: %.2f RMSE' % (train_score))
    # test_score = mean_squared_error(test_y, test_predict)
    # print('Test Score: %.2f RMSE' % (test_score))
    # pdb.set_trace()
    # Gpa = Gauss_s(train_predict[0], test_predict, train_score_MC, train_scorevar, test_score_MC, test_scorevar)

    # training_df = pd.DataFrame()
    # testing_df = pd.DataFrame()
    # training_truth_df = pd.DataFrame()
    # testing_truth_df = pd.DataFrame()
    # training_df['value'] = train_predict[0]
    # training_df['index'] = training_df.index
    # training_df['source'] = 'Training Prediction'

    # testing_df['value'] = test_predict
    # testing_df['index'] = testing_df.index
    # testing_df['source'] = 'Test Prediction'

    # training_truth_df['value'] = train_y[0]
    # training_truth_df['index'] = training_truth_df.index
    # training_truth_df['source'] = 'True Value Training'
    # testing_truth_df['value'] = test_y
    # testing_truth_df['index'] = testing_truth_df.index
    # testing_truth_df['source'] = 'True Value Testing'
    # pdb.set_trace()

    # evaluation = pd.concat([training_df,
    #                       testing_df,
    #                      training_truth_df,
    #                     testing_truth_df
    #                    ], axis=0)

    # predictedanomaly = bounds_df.index[~bounds_df['contained']]
    # pdb.set_trace()

    # figure(test_uncertainty_df, max_pred, min_pred, test_predict, test_y)

    N = 3
    newarr = []
    predictedanomaly = predictedanomaly.sort_values()

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2] and predictedanomaly[i + 2] + 1 == predictedanomaly[i + 3]):
            newarr.append(predictedanomaly[i])

    predicteddanomaly = list(set(newarr))

    predicter = list(range(len(test_uncertainty_df)))

    precision, recall, Accuracy, F1 = score(label_data, predicteddanomaly, test_predict)
    # model.train_score = train_score
    # model.test_score = test_score

    # variance_rf = np.array(test_uncertainty_df['upper_bound']- test_uncertainty_df['lower_bound']) / 3.0
    model.variance = vari
    model.anomalies = predictedanomaly
    model.test_predict = test_predict
    model.train_predict = train_predict
    model.train_time = train_time
    model.test_time = test_time

    return model, train_predict, test_predict, vari, predicteddanomaly, precision, \
           recall, Accuracy, F1


def evaluate_rf(model, train_x, test_x, train_y, test_y, scaler, label_data, pca_variance, pca_var_ratio):
    """
    Evaluates the Random Forest with training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :label_data:
    :return:
    """
    test_uncertainty_df = pd.DataFrame()

    # Forecast

    # train_x = [train_x[i][0] for i in range(train_x.shape[0])]
    # train_x=  np.array(train_x).reshape(-1,1)
    train_x = np.array(train_x)
    # test_x = [test_x[i][0] for i in range(test_x.shape[0])]
    test_x = np.array(test_x)  # .reshape(-1,1)
    # train_y = np.array(train_y).reshape(-1, 1)
    # test_y = np.array(test_y).reshape(-1, 1)

    # pdb.set_trace()
    start_training = time.time()

    # mpg y_hat)
    vari = []
    train_prob_scores_total = []
    test_prob_scores_total = []
    test_uncertainty_mean = []
    test_uncertainty_std = []
    train_scores_total = []
    test_scores_total = []
    train_predict = []
    test_predict = []

    truth_uncertainty_df = pd.DataFrame()
    if train_x.shape[2] > 1:
        for k in range(train_x.shape[2]):
            print(k)
            # pdb.set_trace()
            # nsamples, nx = train_x[:, k, :].shape

            train_x1 = train_x[:, :, k]  # .reshape((nsamples, nx))

            # nsampless, nx = train_y[:, k].shape

            train_y1 = train_y[:, :, k]  # .reshape((nsampless, nx))

            # nsamples1, nx1 = test_x[:, k, :].shape

            test_x1 = test_x[:, :, k]  # .reshape((nsamples1, nx1))
            # nsamples1, nx1 = test_y[:, k].shape

            test_y1 = test_y[:, :, k]  # .reshape((nsamples1, nx1))
            model.fit(train_x1, train_y1)
            end_training = time.time()
            train_time = (end_training - start_training)

            train_predict1 = model.predict(train_x1)  # .reshape(-1,1) #mpg x_hat)
            test_predict1 = model.predict(test_x1)  # .reshape(-1,1)

            train_predict1 = scaler.inverse_transform(train_predict1.reshape(-1, 1))
            train_y1 = scaler.inverse_transform(train_y1.reshape(-1, 1))
            test_predict1 = scaler.inverse_transform(test_predict1.reshape(-1, 1))
            test_y1 = scaler.inverse_transform(test_y1.reshape(-1, 1))
            # Calculate RMSE for train and test

            n_experiments = 3
            test_uncertainty_df = pd.DataFrame()

            ##%
            test_time = time.time()

            for i in tqdm(range(n_experiments)):
                # model1.fit(train_x, train_y)
                # model1 = RandomForestRegressor(n_estimators=random.randint(50, 1000), criterion='mse', max_depth=None,
                #                              min_samples_split=random.randint(2, 4),
                #                             min_samples_leaf=random.randint(2, 4), max_features='auto', max_leaf_nodes=None,
                #                            bootstrap=2,
                #                           oob_score=False, n_jobs=random.randint(2,4), random_state=None, verbose=0)

                I = np.random.choice(train_x1.shape[0], random.randint(train_x1.shape[0] - random.randint(0, 200),
                                                                       train_x1.shape[0] + random.randint(0, 200)),
                                     replace=True)
                np.random.shuffle(I)
                # J= np.random.choice(test_x.shape[0], random.randint(test_x.shape[0] , test_x.shape[0]), replace=True)
                model.fit(train_x1[I[:train_x1.shape[0]]], train_y1[I[:train_y1.shape[0]]])
                bootstrap_train_x = train_x1[I[:train_x1.shape[0]]]  # train_x[I]
                bootstrap_train_y = train_y1[I[:train_y1.shape[0]]]  # train_y[I]
                bootstrap_train_x = np.array(bootstrap_train_x)
                bootstrap_train_y = np.array(bootstrap_train_y)
                model.fit(bootstrap_train_x, bootstrap_train_y)
                bootstrap_test_x = test_x1
                bootstrap_test_y = test_y1
                bootstrap_test_x = np.array(bootstrap_test_x)
                bootstrap_test_y = np.array(bootstrap_test_y)
                bootstrap_test_predict = model.predict(bootstrap_test_x)
                bootstrap_test_predict = model.predict(test_x1)
                # bootstrap_test_predict = scaler.inverse_transform([bootstrap_test_predict])[0]
                # bootstrap_test_y = scaler.inverse_transform([bootstrap_test_y])[0]
                bootstrap_test_rmse = np.sqrt(mean_squared_error(bootstrap_test_predict, bootstrap_test_y))

                test_uncertainty_df['value_{}'.format(i)] = bootstrap_test_predict
            test_time = (time.time() - test_time) / n_experiments

            #    test_uncertainty_df['value'] = test_predict.squeeze().tolist()

            idx = list(range(train_x.shape[0]))
            # shuffle the data
            np.random.shuffle(idx)
            # pdb.set_trace()
            # model1 = RandomForestRegressor(n_estimators=model.n_estimators, min_samples_leaf=4, max_depth=None)
            model.fit(train_x1[idx[:train_x1.shape[0]]], train_y1[idx[:train_y1.shape[0]]])

            # idx = list(range(test_x.shape[0]))
            # shuffle the data
            # np.random.shuffle(idx)
            # pdb.set_trace()

            err_down, err_up = pred_ints(model, test_x1)

            test_uncertainty_df['value_mean'] = test_uncertainty_df.filter(like='value', axis=1).mean(axis=1)
            test_uncertainty_df['value_std'] = test_uncertainty_df.filter(like='value', axis=1).std(axis=1)

            test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - 3 * abs(
                test_uncertainty_df['value_std'])
            test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + 3 * abs(
                test_uncertainty_df['value_std'])

            # pdb.set_trace()
            test_uncertainty_df['lower_bound'] = err_down
            test_uncertainty_df['upper_bound'] = err_up
            # pdb.set_trace()
            truth_uncertainty_df['value'] = test_y1[:, 0]
            truth_uncertainty_df['index'] = truth_uncertainty_df.index
            # pdb.set_trace()
            # test_predict = scaler.inverse_transform(test_predict)

            #  test_y = scaler.inverse_transform([test_y])
            # Calculate RMSE for train and test
            #    train_score = mean_squared_error(train_y[:, 0], train_predict[:, 0])
            train_scores = np.sqrt((train_y1.squeeze() - train_predict1.squeeze()) ** 2)
            test_scores = np.sqrt((test_y1.squeeze() - test_predict1.squeeze()) ** 2)
            #     test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
            # pdb.set_trace()

            # train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
            # test_score_MC = mean_squared_error(test_y[0], test_MC_predict_point[0, :])
            # print('Train Score: %.2f RMSE' % (train_score))
            #   test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
            # print('Test Score: %.2f RMSE' % (test_score))
            # pdb.set_trace()
            train_score_mean = np.mean(train_scores)
            train_score_std = np.std(train_scores)
            test_score_mean = np.mean(test_scores)
            test_score_std = np.std(test_scores)

            train_prob_scores, test_prob_scores = Gauss_s(train_scores, test_scores, train_predict1, test_predict1,
                                                          train_score_mean, train_score_std,
                                                          test_score_mean, test_score_std)

            # pdb.set_trace()
            vari.append(test_uncertainty_df['value_std'])
            train_prob_scores_total.append(train_prob_scores)
            test_prob_scores_total.append(test_prob_scores)
            test_uncertainty_mean.append(test_uncertainty_df['value_mean'])
            test_uncertainty_std.append(test_uncertainty_df['value_std'])
            train_scores_total.append(train_scores)
            test_scores_total.append(test_scores)
            train_predict.append(train_predict1)
            test_predict.append(test_predict1)

        train_ano_scores = np.sum(train_prob_scores_total, axis=0)
        test_ano_scores = np.sum(test_prob_scores_total, axis=0)
        #  pdb.set_trace()
        _, anomaly_score_eig, th_eig = eig_method(train_prob_scores_total, test_prob_scores_total, pca_var_ratio, vari)

        # pdb.set_trace()

        test_uncertainty_df['lower_bound'] = -  th_eig

        test_uncertainty_df['upper_bound'] = th_eig
        bounds_df = pd.DataFrame()

        bounds_df['lower_bound'] = test_uncertainty_df['lower_bound']
        # bounds_df['prediction'] = test_uncertainty_df['value_mean']
        # bounds_df['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df['upper_bound'] = test_uncertainty_df['upper_bound']
        # pdb.set_trace()

        bounds_df['contained'] = (bounds_df['upper_bound'] >= anomaly_score_eig.squeeze())

        print("Proportion of points contained within 99% confidence interval:",
              bounds_df['contained'].mean())
        # pdb.set_trace()
        predictedanomaly = bounds_df.index[~bounds_df['contained']]
        # pdb.set_trace()

        # evbus = varU(train_x, train_y, test_x)

        # v = evbus.calculate_variance()

        # test_uncertainty_df['index'] = pd.DataFrame(test_predict).index
        # import plotly.graph_objects as go
        # test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
        # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
        # truth_uncertainty_plot_df = pd.DataFrame()

        # truth_uncertainty_plot_df['value'] = test_y[:, k]
        # truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

        # .copy(deep=True)
        # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

        # upper_trace = go.Scatter(
        #   x=test_uncertainty_plot_df['index'],
        #  y=test_uncertainty_plot_df['upper_bound'],
        # mode='lines',
        # fill=None,
        # name='99% Upper Confidence Bound 3xSTD',
        # )
        # lower_trace = go.Scatter(
        #   x=test_uncertainty_plot_df['index'],
        #  y=test_uncertainty_plot_df['lower_bound'],
        # mode='lines',
        # fill='tonexty',
        # name='99% Lower Confidence Bound 3xSTD',
        # fillcolor='rgba(255, 211, 0, 0.1)',
        # )
        # real_trace = go.Scatter(
        #   x=truth_uncertainty_plot_df['index'],
        #  y=truth_uncertainty_plot_df['value'],
        # mode='lines',
        # fill=None,
        # name='Real Values'
        # )

    # data = [upper_trace, lower_trace, real_trace]

    # fig = go.Figure(data=data)
    # fig.update_layout(title='RF Uncertainty',
    #                 xaxis_title='index',
    #                yaxis_title='value',
    #               legend_font_size=14,
    # paper_bgcolor='rgba(0,0,0,0)',
    # plot_bgcolor ='rgba(0,0,0,0)',
    #              )
    # fig.show()
    # bounds_df = pd.DataFrame()
    # pdb.set_trace()

    # Using 99% confidence bounds

    # bounds_df['lower_bound'] = list(map(lambda x: max(x, -1), list(test_uncertainty_plot_df['lower_bound'])))
    # bounds_df['prediction'] = test_uncertainty_plot_df['value_mean']
    # bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    # bounds_df['upper_bound'] = list(map(lambda x: min(x, 1), list(test_uncertainty_plot_df['upper_bound'])))

    # bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
    #                         (bounds_df['real_value'] <= bounds_df['upper_bound']))

    # print("Proportion of points contained within 99% confidence interval:",
    #      bounds_df['contained'].mean())
    # predictedanomaly = bounds_df.index[~bounds_df['contained']]

    # model_n = IntervalRegressor(RandomForestRegressor(n_estimators=50))
    # model_n.fit(train_x, train_y)
    # pred_y = model_n.predict(test_x)
    # pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
    # test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
    # bootstrapped_pred = model_n.predict(test_x)
    # min_pred = bootstrapped_pred[:, 0]
    # max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    # pdb.set_trace()

    # train_y = scaler.inverse_transform(train_y)
    # test_predict = scaler.inverse_transform(test_predict)
    # test_y = scaler.inverse_transform(test_y)
    predictedanomaly = predictedanomaly.sort_values()

    N = 3
    newarr = []

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2]):  # and predictedanomaly[i + 3] + 1 == predictedanomaly[i + 4] and predictedanomaly[i + 4] + 1 ==
            # predictedanomaly[i + 5]):
            newarr.append(predictedanomaly[i])

    predicteddanomaly = list(set(newarr))

    # realanomaly = label_data['index']

    predicter = list(range(len(test_uncertainty_df)))
    # pdb.set_trace()
    precision, recall, Accuracy, F1 = score(label_data, predictedanomaly, test_predict)

    # plt.errorbar(test_y, test_predict, yerr=np.sqrt(variance_rf.mean()), fmt='o')
    #   pdb.set_trace()

    # train_score = mean_squared_error(train_y.reshape(1, -1)[0], train_predict[0])
    # print('Train Score: %.2f RMSE' % (train_score))
    # test_score = mean_squared_error(test_y.reshape(1, -1)[0], test_predict[0])
    # print('Test Score: %.2f RMSE' % (test_score))
    # model.train_score = train_score
    # model.test_score = test_score

    variance_rf = vari  # (np.array(err_up) - np.array(err_down)) / 6.0
    # variance_rf = np.array(test_uncertainty_df['upper_bound']- test_uncertainty_df['lower_bound']) / 3.0
    model.variance = variance_rf
    model.anomalies = predictedanomaly
    model.test_predict = test_predict
    model.train_predict = train_predict
    # pdb.set_trace()
    model.train_time = train_time
    model.test_time = test_time

    return model, train_predict, test_predict, variance_rf, predictedanomaly, precision, recall, Accuracy, F1


def evaluate_rnn(model, train_x, test_x, train_y, test_y, scaler, optimiser, name, label_data, pca_variance,
                 pca_var_ratio):
    """
    Evaluates the RNN model using the training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :param optimiser:
    :return:
    """
    model.compile(loss='mean_squared_error', optimizer='Adam')
    t1 = time.time()
    # pdb.set_trace()
    # [samples, timesteps, features]

    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[2], train_x.shape[1]))
    train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[2]))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[2], test_x.shape[1]))
    test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[2]))
    print(train_x.shape)
    print(train_y.shape)
    pdb.set_trace()

    model.fit(train_x, train_y, epochs=rnn_epochs, verbose=2)
    # Forecast

    train_time = time.time() - t1
    print('End of training RNN')
    # pdb.set_trace()

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts
    # pdb.set_trace()
    print('start MC Dropout distribution predict')

    # train_MC_predict = predict_dist(train_x, model, num_samples=100)

    # test_MC_predict = predict_dist(test_x, model, num_samples=100)
    print('End MC Dropoutdistribution predict')
    print('start MC Dropout predict point')

    #
    n_experiments = 20
    print('End MC Dropout predict point')
    print('start MC Dropout')
    test_uncertainty_df = pd.DataFrame()
    # test_uncertainty_df['date'] = testing_df['date']
    indices = [i for i in range(len(test_y))]

    test_uncertainty_df['index'] = indices
    # for i in range(n_experiments):
    t2 = time.time()
    print()
    pdb.set_trace()

    experiment_predictions = np.stack([model(test_x, training=True) for _ in range(mcdropout_samples)])

    test_time = (time.time() - t2) / rnn_epochs
    # pdb.set_trace()
    print('End MC Dropout predict point')
    print('start variance Calculations')
    test_uncertainty_dff = {}
    true_uncertainty_dff = {}
    test_uncertainty_dfft = []
    # pdb.set_trace()
    # pca
    #  scaler = MinMaxScaler(feature_range=(-1, 1))

    #    experiment_predictions_1 = scaler.fit_transform(
    #       experiment_predictions.mean(axis=0))
    vari = []
    train_prob_scores_total = []
    test_prob_scores_total = []
    test_uncertainty_mean = []
    test_uncertainty_std = []
    train_scores_total = []
    test_scores_total = []
    # pdb.set_trace()
    # pca = PCA(n_components=0.99, svd_solver='full')

    # pca.fit(experiment_predictions1)

    # experiment_predictions_reduced = pca.transform(experiment_predictions1)
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    M = experiment_predictions.mean(axis=0).shape[1]

    if train_x.shape[1] > 1:
        for i in range(M):
            print(i)
            test_uncertainty_df['value_mean'] = experiment_predictions.mean(axis=0)[:, i]
            test_uncertainty_df['value_mean'] = scaler.fit_transform(
                test_uncertainty_df['value_mean'].values.reshape(-1, 1))
            # scaler.inverse_transform(test_uncertainty_df['value_mean'].values.reshape(1, -1))
            test_uncertainty_df['value_mean'] = scaler.fit_transform(
                test_uncertainty_df['value_mean'].values.reshape(-1, 1))

            test_uncertainty_df['value_mean'] = scaler.inverse_transform(
                test_uncertainty_df['value_mean'].values.reshape(-1, 1))
            test_uncertainty_df['value_std'] = experiment_predictions.std(axis=0)[:, i]
            test_uncertainty_df = test_uncertainty_df[['index', 'value_mean', 'value_std']]

            # pdb.set_trace()
            # train_MC_predict_point, train_MC_predict_var = predict_point(train_x, model, num_samples=mcdropout_samples)
            # test_MC_predict_point, test_MC_predict_var = predict_point(test_x, model, num_samples=mcdropout_samples)
            # test_MC_predict_var = test_uncertainty_df['value_std']
            # test_MC_predict_point = test_uncertainty_df['value_mean']
            # pdb.set_trace()

            train_predict1 = scaler.inverse_transform(train_predict[:, i].reshape(-1, 1))
            train_y1 = scaler.inverse_transform(train_y[:, i].reshape(-1, 1))
            test_y1 = scaler.inverse_transform(test_y[:, i].reshape(-1, 1))
            test_predict1 = scaler.inverse_transform(test_predict[:, i].reshape(-1, 1))

            # train_MC_predict_point = scaler.inverse_transform([train_MC_predict_point])
            # test_MC_predict_point = scaler.inverse_transform([test_MC_predict_point])
            # test_uncertainty_df = pd.DataFrame()
            #    pdb.set_trace()
            # train_MC_predict = train_MC_predict_point
            # test_MC_predict = test_MC_predict_point

            # test_uncertainty_df = test_uncertainty_df[['index',  'value_mean', 'value_std']]
            # test_uncertainty_df['value_mean']=test_MC_predict_point[0]
            # test_uncertainty_df['lower_bound'] = test_MC_predict_point[0] - 3 * abs(test_MC_predict_var)
            # test_uncertainty_df['upper_bound'] = test_MC_predict_point[0] + 3 * abs(test_MC_predict_var)
            test_uncertainty_df['index'] = test_uncertainty_df.index
            import plotly.graph_objects as go

            test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
            # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
            truth_uncertainty_plot_df = pd.DataFrame()

            truth_uncertainty_plot_df['value'] = test_y[:, i]
            truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index
            # pdb.set_trace()
            # test_predict = scaler.inverse_transform(test_predict)

            #  test_y = scaler.inverse_transform([test_y])
            # Calculate RMSE for train and test
            #    train_score = mean_squared_error(train_y[:, 0], train_predict[:, 0])
            train_scores = np.sqrt((train_y1.squeeze() - train_predict1.squeeze()) ** 2)
            test_scores = np.sqrt((test_y1.squeeze() - test_predict1.squeeze()) ** 2)
            #     test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
            # pdb.set_trace()

            # train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
            # test_score_MC = mean_squared_error(test_y[0], test_MC_predict_point[0, :])
            # print('Train Score: %.2f RMSE' % (train_score))
            #   test_score = mean_squared_error(test_y[:, 0], test_predict[:, 0])
            # print('Test Score: %.2f RMSE' % (test_score))
            # pdb.set_trace()
            train_score_mean = np.mean(train_scores)
            train_score_std = np.std(train_scores)
            test_score_mean = np.mean(test_scores)
            test_score_std = np.std(test_scores)

            train_prob_scores, test_prob_scores = Gauss_s(train_scores, test_scores, train_predict, test_predict,
                                                          train_score_mean, train_score_std,
                                                          test_score_mean, test_score_std)

            # pdb.set_trace()
            vari.append(test_uncertainty_df['value_std'])
            train_prob_scores_total.append(train_prob_scores)
            test_prob_scores_total.append(test_prob_scores)
            test_uncertainty_mean.append(test_uncertainty_df['value_mean'])
            test_uncertainty_std.append(test_uncertainty_df['value_std'])
            train_scores_total.append(train_scores)
            test_scores_total.append(test_scores)

        train_ano_scores = np.sum(train_prob_scores_total, axis=0)
        test_ano_scores = np.sum(test_prob_scores_total, axis=0)
        _, anomaly_score_eig, th_eig = eig_method(train_prob_scores_total, test_prob_scores_total, pca_var_ratio, vari)

        # pdb.set_trace()

        test_uncertainty_df['lower_bound'] = -  th_eig

        test_uncertainty_df['upper_bound'] = th_eig
        bounds_df = pd.DataFrame()

        bounds_df['lower_bound'] = test_uncertainty_df['lower_bound']
        # bounds_df['prediction'] = test_uncertainty_df['value_mean']
        # bounds_df['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df['upper_bound'] = test_uncertainty_df['upper_bound']
        # pdb.set_trace()

        bounds_df['contained'] = (bounds_df['upper_bound'] >= anomaly_score_eig.squeeze())

        print("Proportion of points contained within 99% confidence interval:",
              bounds_df['contained'].mean())
        # pdb.set_trace()
        predictedanomaly = bounds_df.index[~bounds_df['contained']]
        # pdb.set_trace()

        model.train_score = train_scores_total
        model.test_score = test_scores_total
        # model.train_score_MC = train_score_MC
        # model.test_score_MC = test_score_MC
        training_df = pd.DataFrame()
        testing_df = pd.DataFrame()
        training_truth_df = pd.DataFrame()
        testing_truth_df = pd.DataFrame()

        #        training_df['value'] = train_MC_predict_point[0]
        #        training_df['index'] = training_df.index
        #       training_df['source'] = 'Training Prediction'
        #       testing_df['value'] = test_MC_predict_point[0]
        #      testing_df['index'] = testing_df.index
        #     testing_df['source'] = 'Test Prediction'
        #    training_truth_df['value'] = train_y[0]
        #    training_truth_df['index'] = training_truth_df.index
        #    training_truth_df['source'] = 'True Value Training'
        #    testing_truth_df['value'] = test_y[0]
        #    testing_truth_df['index'] = testing_truth_df.index
        #    testing_truth_df['source'] = 'True Value Testing'
        # pdb.set_trace()

        #   evaluation = pd.concat([training_df,
        #                          testing_df,
        #                         training_truth_df,
        #                        testing_truth_df
        #                       ], axis=0)

        predictedanomaly = bounds_df.index[~bounds_df['contained']]
        # N = Nmax_gen(predictedanomaly, test_uncertainty_df, truth_uncertainty_plot_df, label_data, name)

        N = 3
        newarr = []
        predictedanomaly = predictedanomaly.sort_values()

        for i in range(len(predictedanomaly) - N):
            if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                    predictedanomaly[
                        i + 2]):
                newarr.append(predictedanomaly[i])

        predicteddanomaly = list(set(newarr))

        predicter = list(range(len(test_uncertainty_df)))
        # pdb.set_trace()

        precision, recall, Accuracy, F1 = score(label_data, predicteddanomaly, test_predict)
        # pdb.set_trace()

        model.train_score_MC = train_scores
        model.test_score_MC = test_scores
        model.train_score = train_scores_total
        model.test_score = test_scores_total
        model.train_predict = train_predict
        model.test_predict = test_predict
        model.test_MC_predict = test_predict
        model.test_MC_predict_point = test_predict
        model.train_MC_predict = train_predict
        model.train_MC_predict_point = train_predict
        model.test_MC_predict_var = vari
        model.train_MC_predict_var = vari
        model.predictedanomaly = predictedanomaly
        model.train_time = train_time
        model.test_time = test_time
        tensorflow.compat.v1.trainable_variables(
            scope=None
        )
        model.trainable_parameters = np.sum(
            [np.prod(v.get_shape().as_list()) for v in tensorflow.compat.v1.trainable_variables()])

        return model, train_scores, test_scores, train_scores, test_scores, train_predict, test_predict, test_predict, test_predict, train_predict, train_predict, vari, vari, predictedanomaly, precision, recall, Accuracy, F1
