import numpy as np
import pandas as pd

from src.utils import Gauss_s
from src.utils import eig_method
from src.utils import score


def ensemble_stacking(label_data, rnn_model, lr_model, rf_model, model_1_values, model_3_values, model_2_values,
                      test, scaler, train_x=None, train_y=None, var_pca=None, var_pca_ratio=None, *args, **kwargs):
    """
    Ensemble result of 2 models using stacking and averaging.
    Takes both model predictions, averages them and calculates the new RMSE
    rnn_model - model 1
    lr_model - model 3
    rf_model - model 2
    :param model_1_values:
    :param model_2_values:
    :param model_3_values:
    :return:
    """

    if lr_model is not None:
        var_rnn = rnn_model.test_MC_predict_var
        var_rf = rf_model.variance
        var_lr = lr_model.variance
        model_3_values = np.array(model_3_values)
        model_3_values = model_3_values.reshape(model_3_values.shape[1], model_3_values.shape[0])
        model_3_values = model_3_values.squeeze()
        model_1_values = model_1_values.squeeze()
        model_2_values = np.array(model_2_values)
        model_2_values = model_2_values.reshape(model_2_values.shape[1], model_2_values.shape[0])
        model_2_values = model_2_values.squeeze()
        # Generates the stacking values by averaging both predictions
        # model1 rnn   model2 rf    model3 lr
        stacking_values12 = []
        stacking_values13 = []
        stacking_values23 = []
        stacking_values123 = []
        var12 = []
        var13 = []
        var23 = []
        var_total = []
        var_total_new = []
        var13_new = []
        var23_new = []
        var12_new = []
        var123_new = []
        # pdb.set_trace()
        var_rnn = np.array(var_rnn)
        var_rnn = var_rnn.reshape(var_rnn.shape[1], var_rnn.shape[0])
        var_rf = np.array(var_rf)
        var_rf = var_rf.reshape(var_rf.shape[1], var_rf.shape[0])
        var_lr = np.array(var_lr)
        var_lr = var_lr.reshape(var_lr.shape[1], var_lr.shape[0])
        test_uncertainty_mean = []
        test_uncertainty_std = []

        train_prob_scores_total = []
        test_prob_scores_total = []
        train_scores_total = []
        test_scores_total = []
        vari = []

        for k in range(model_1_values.shape[1]):
            predicted12 = []  # np.array([])
            predicted13 = []  # np.array([])
            predicted23 = []  # np.array([])
            predicted123 = []  # np.array([])
            var12 = []  # np.array([])
            var13 = []  # np.array([])
            var23 = []  # np.array([])
            var_total = []  # np.array([])
            var_total_new = []  # np.array([])
            var13_new = []  # np.array([])
            var23_new = []  # np.array([])
            var12_new = []  # np.array([])
            var123_new = []  # np.array([])

            stacking_values123 = []
            stacking_values12 = []
            stacking_values13 = []
            stacking_values23 = []

            for i in range(model_1_values.shape[0]):
                #   print(i)
                w_rnn = abs(1 / var_rnn[i, k])
                w_lr = abs(1 / var_lr[i, k])
                w_rf = abs(1 / var_rf[i, k])
                temp123 = (w_rnn * model_1_values[i, k] + w_lr * model_3_values[i, k] + w_rf * model_2_values[i, k]) / (
                        w_rf + w_lr + w_rnn)

                # stacking_values.append((model_1_values[i][0] + model_2_values[0][i]) / 2)
                stacking_values123.append(temp123)
                temp13 = (w_rnn * model_1_values[i, k] + w_lr * model_3_values[i, k]) / (w_rnn + w_lr)
                stacking_values13.append(temp13)
                temp12 = (model_1_values[i, k] * w_rnn + model_2_values[i, k] * w_rf) / (w_rnn + w_rf)
                stacking_values12.append(temp12)
                temp23 = (model_2_values[i, k] * w_rf + model_3_values[i, k] * w_lr) / (w_lr + w_rf)
                stacking_values23.append(temp23)
                cov12 = np.cov(model_1_values[i, k], model_2_values[i, k])
                cov13 = np.cov(model_1_values[i, k], model_3_values[i, k])
                cov23 = np.cov(model_2_values[i, k], model_3_values[i, k])
                cov123 = np.cov(model_1_values[i, k], model_2_values[i, k], model_3_values[i, k])
                predicted12 = np.append(predicted12, temp12)
                predicted13 = np.append(predicted13, temp13)
                predicted23 = np.append(predicted23, temp23)
                predicted123 = np.append(predicted123, temp123)

                var_total.append(1 / (w_rf + w_lr + w_rnn))
                var_total_new.append(
                    1 / (w_rf + w_lr + w_rnn) + (2 / (w_rf + w_lr + w_rnn) ** 2) * (cov12 + cov13 + cov23))

                var13.append(1 / (w_rnn + w_lr))
                var12.append(1 / (w_rnn + w_rf))
                var23.append(1 / (w_rf + w_lr))
                var13_new.append(1 / (w_rnn + w_lr) + (2 / (w_rnn + w_lr) ** 2) * cov13)
                var12_new.append(1 / (w_rnn + w_rf) + (2 / (w_rnn + w_rf) ** 2) * cov12)
                var23_new.append(1 / (w_rf + w_lr) + (2 / (w_rf + w_lr) ** 2) * cov23)

                # pdb.set_trace()

            # Calculates the new RMSE
            # pdb.set_trace()
            test1 = scaler.inverse_transform(test[:, :, k])
            # Calculates the new RMSE
            n_values = min(len(test), len(model_1_values))

            # stacking_values12 = np.nan_to_num(stacking_values12)
            # stacking_values23 = np.nan_to_num(stacking_values23)
            # stacking_values13 = np.nan_to_num(stacking_values13)
            # stacking_values123 = np.nan_to_num(stacking_values123)

            # stacking_values_uncertainty = stacking_values123
            rmse12 = ((test1.squeeze() - predicted12) ** 2).mean()
            rmse13 = ((test1.squeeze() - predicted13) ** 2).mean()
            rmse23 = ((test1.squeeze() - predicted23) ** 2).mean()
            rmse123 = ((test1.squeeze() - predicted123) ** 2).mean()
            test_uncertainty_df12 = pd.DataFrame()
            test_uncertainty_df13 = pd.DataFrame()
            test_uncertainty_df23 = pd.DataFrame()
            test_uncertainty_df123 = pd.DataFrame()
            # pdb.set_trace()
            # var_total = np.array(var_total)

            test_uncertainty_df123['lower_bound'] = np.array(stacking_values123) - 3 * np.array(var_total)
            test_uncertainty_df123['upper_bound'] = np.array(stacking_values123) + 3 * np.array(var_total)
            # test_uncertainty_df123['index'] = pd.DataFrame(test1).index.values

            test_uncertainty_df12['lower_bound'] = np.array(stacking_values12) - 3 * np.array(var12)
            test_uncertainty_df12['upper_bound'] = np.array(stacking_values12) + 3 * np.array(var12)
            # pdb.set_trace()
            # test_uncertainty_df12['index'] = pd.DataFrame(test1).index.values
            test_uncertainty_df13['lower_bound'] = np.array(stacking_values13) - 3 * np.array(var13)
            test_uncertainty_df13['upper_bound'] = np.array(stacking_values13) + 3 * np.array(var13)
            # test_uncertainty_df13['index'] = pd.DataFrame(test1).index.values
            test_uncertainty_df23['lower_bound'] = np.array(stacking_values23) - 3 * np.array(var23)
            test_uncertainty_df23['upper_bound'] = np.array(stacking_values23) + 3 * np.array(var23)
            # test_uncertainty_df23['index'] = pd.DataFrame(test1).index.values
            # test_uncertainty_df12.set_index('index', inplace=True)
            # test_uncertainty_df13.set_index('index', inplace=True)
            # test_uncertainty_df23.set_index('index', inplace=True)
            test_uncertainty_plot_df123 = test_uncertainty_df123  # .copy(deep=True)
            test_uncertainty_plot_df12 = test_uncertainty_df12  # .copy(deep=True)
            test_uncertainty_plot_df13 = test_uncertainty_df13  # .copy(deep=True)
            test_uncertainty_plot_df23 = test_uncertainty_df23  # .copy(deep=True)

            truth_uncertainty_plot_df = pd.DataFrame()

            truth_uncertainty_plot_df['value'] = test1.squeeze()  # [:, k]
            truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index
            # pdb.set_trace()
            # test_predict = scaler.inverse_transform(test_predict)

            #  test_y = scaler.inverse_transform([test_y])
            # Calculate RMSE for train and test
            train_y1 = train_y[:, :, k]
            train_y1 = scaler.inverse_transform(train_y1)
            #    train_score = mean_squared_error(train_y[:, 0], train_predict[:, 0])
            train_scores = np.sqrt((train_y1.squeeze() - stacking_values123[k].squeeze()) ** 2)
            test_scores = np.sqrt((test1.squeeze() - stacking_values123[k].squeeze()) ** 2)
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

            train_prob_scores, test_prob_scores = Gauss_s(train_scores, test_scores, train_y1, stacking_values123,
                                                          train_score_mean, train_score_std,
                                                          test_score_mean, test_score_std)
            var_total1 = np.array(var_total)

            # pdb.set_trace()
            vari.append(var_total1)
            train_prob_scores_total.append(train_prob_scores)
            test_prob_scores_total.append(test_prob_scores)
            test_uncertainty_mean.append(np.array(stacking_values123))
            test_uncertainty_std.append(var_total)
            train_scores_total.append(train_scores)
            test_scores_total.append(test_scores)
            # stacking_values123=list(stacking_values123)

    train_ano_scores = np.sum(train_prob_scores_total, axis=0)
    test_ano_scores = np.sum(test_prob_scores_total, axis=0)

    _, anomaly_score_eig, th_eig = eig_method(train_prob_scores_total, test_prob_scores_total, var_pca_ratio, vari)

    # pdb.set_trace()

    test_uncertainty_df12['lower_bound'] = -  th_eig

    test_uncertainty_df12['upper_bound'] = th_eig
    bounds_df = pd.DataFrame()

    bounds_df['lower_bound'] = test_uncertainty_df12['lower_bound']
    # bounds_df['prediction'] = test_uncertainty_df['value_mean']
    # bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df['upper_bound'] = test_uncertainty_df12['upper_bound']
    # pdb.set_trace()

    bounds_df['contained'] = (bounds_df['upper_bound'] >= anomaly_score_eig.squeeze())

    print("Proportion of points contained within 99% confidence interval:",
          bounds_df['contained'].mean())
    # pdb.set_trace()
    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    # pdb.set_trace()

    # model.train_score = train_scores_total
    # model.test_score = test_scores_total
    # model.train_score_MC = train_score_MC
    # model.test_score_MC = test_score_MC
    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()
    training_truth_df = pd.DataFrame()
    testing_truth_df = pd.DataFrame()

    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    # truth_uncertainty_plot_df = pd.DataFrame()
    # truth_uncertainty_plot_df['value'] = test[0]
    # truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

    # .copy(deep=True)
    # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

    # upper_trace = go.Scatter(
    #       x=test_uncertainty_plot_df['index'],
    #      y=test_uncertainty_plot_df['upper_bound'],
    #     mode='lines',
    #    fill=None,
    #   name='99% Upper Confidence Bound   '
    # )
    # lower_trace = go.Scatter(
    #       x=test_uncertainty_plot_df['index'],
    #      y=test_uncertainty_plot_df['lower_bound'],
    #     mode='lines',
    #    fill='tonexty',
    #   name='99% Lower Confidence Bound',
    #  fillcolor='rgba(255, 211, 0, 0.1)',
    # )
    # real_trace = go.Scatter(
    #       x=truth_uncertainty_plot_df['index'],
    #      y=truth_uncertainty_plot_df['value'],
    #     mode='lines',
    #    fill=None,
    #   name='Real Values'
    # )

    # data = [upper_trace, lower_trace, real_trace]

    # fig = go.Figure(data=data)
    # fig.update_layout(title='RF Uncertainty',
    #                      xaxis_title='index',
    #                     yaxis_title='value',
    #                    legend_font_size=14,
    #                   )
    # fig.show()
    bounds_df123 = pd.DataFrame()
    bounds_df12 = pd.DataFrame()
    bounds_df13 = pd.DataFrame()
    bounds_df23 = pd.DataFrame()
    bounds_df123['lower_bound'] = test_uncertainty_plot_df123['lower_bound']
    bounds_df123['upper_bound'] = test_uncertainty_plot_df123['upper_bound']
    #    bounds_df123['index'] = test_uncertainty_plot_df123['index']
    bounds_df123['prediction'] = stacking_values123
    bounds_df123['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df123['contained'] = ((bounds_df123['real_value'] >= bounds_df123['lower_bound']) &
                                 (bounds_df123['real_value'] <= bounds_df123['upper_bound']))
    predictedanomaly123 = bounds_df123.index[~bounds_df123['contained']]

    bounds_df12['lower_bound'] = test_uncertainty_plot_df12['lower_bound']
    bounds_df12['upper_bound'] = test_uncertainty_plot_df12['upper_bound']
    #   bounds_df12['index'] = test_uncertainty_plot_df12['index']
    bounds_df12['prediction'] = stacking_values12
    bounds_df12['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df12['contained'] = ((bounds_df12['real_value'] >= bounds_df12['lower_bound']) & (
            bounds_df12['real_value'] <= bounds_df12['upper_bound']))
    predictedanomaly12 = bounds_df12.index[~bounds_df12['contained']]
    bounds_df13['lower_bound'] = test_uncertainty_plot_df13['lower_bound']
    bounds_df13['upper_bound'] = test_uncertainty_plot_df13['upper_bound']
    # bounds_df13['index'] = test_uncertainty_plot_df13['index']
    bounds_df13['prediction'] = stacking_values13
    bounds_df13['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df13['contained'] = ((bounds_df13['real_value'] >= bounds_df13['lower_bound']) & (
            bounds_df13['real_value'] <= bounds_df13['upper_bound']))
    predictedanomaly13 = bounds_df13.index[~bounds_df13['contained']]

    bounds_df23['lower_bound'] = test_uncertainty_plot_df23['lower_bound']
    bounds_df23['upper_bound'] = test_uncertainty_plot_df23['upper_bound']
    # bounds_df23['index'] = test_uncertainty_plot_df23['index']
    bounds_df23['prediction'] = stacking_values23
    bounds_df23['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df23['contained'] = ((bounds_df23['real_value'] >= bounds_df23['lower_bound']) & (
            bounds_df23['real_value'] <= bounds_df23['upper_bound']))
    predictedanomaly23 = bounds_df23.index[~bounds_df23['contained']]

    N = 5
    newarr123 = []
    newarr12 = []
    newarr13 = []
    newarr23 = []

    for i in range(len(predictedanomaly123) - N):
        if (predictedanomaly123[i] + 1 == predictedanomaly123[i + 1] and predictedanomaly123[i + 1] + 1 ==
                predictedanomaly123[
                    i + 2] and predictedanomaly123[i + 3] + 1 == predictedanomaly123[i + 4]):
            newarr123.append(predictedanomaly123[i])

    for i in range(len(predictedanomaly12) - N):
        if (predictedanomaly12[i] + 1 == predictedanomaly12[i + 1] and predictedanomaly12[i + 1] + 1 ==
                predictedanomaly12[
                    i + 2] and predictedanomaly12[i + 3] + 1 == predictedanomaly12[i + 4]):
            newarr12.append(predictedanomaly12[i])

    for i in range(len(predictedanomaly13) - N):
        if (predictedanomaly13[i] + 1 == predictedanomaly13[i + 1] and predictedanomaly13[i + 1] + 1 ==
                predictedanomaly13[
                    i + 2] and predictedanomaly13[i + 3] + 1 == predictedanomaly13[i + 4]):
            newarr13.append(predictedanomaly13[i])

    for i in range(len(predictedanomaly23) - N):
        if (predictedanomaly23[i] + 1 == predictedanomaly23[i + 1] and predictedanomaly23[i + 1] + 1 ==
                predictedanomaly23[
                    i + 2] and predictedanomaly23[i + 3] + 1 == predictedanomaly23[i + 4]):
            newarr23.append(predictedanomaly23[i])

    newarr123 = np.array(newarr123)
    newarr12 = np.array(newarr12)
    newarr13 = np.array(newarr13)
    newarr23 = np.array(newarr23)
    newarr123 = newarr123.astype(int)
    newarr12 = newarr12.astype(int)
    newarr13 = newarr13.astype(int)
    newarr23 = newarr23.astype(int)

    #        newarr.append(predictedanomaly[i + 1])
    #       newarr.append(predictedanomaly[i + 2])

    predicteddanomaly123 = list(set(newarr123))
    predicteddanomaly12 = list(set(newarr12))
    predicteddanomaly13 = list(set(newarr13))
    predicteddanomaly23 = list(set(newarr23))
    # pdb.set_trace()

    # realanomaly = label_data['index']
    predicter = list(range(len(test_uncertainty_plot_df123)))
    precision123, recall123, Accuracy123, F1123 = score(label_data, predicteddanomaly123, stacking_values123)
    precision12, recall12, Accuracy12, F112 = score(label_data, predicteddanomaly12, stacking_values12)
    precision13, recall13, Accuracy13, F113 = score(label_data, predicteddanomaly13, stacking_values13)
    precision23, recall23, Accuracy23, F123 = score(label_data, predicteddanomaly23, stacking_values23)

    # pdb.set_trace()

    #   predicteddanomaly123, stacking_values12, stacking_values13, stacking_values23, stacking_values123, rmse12, rmse13, rmse23, rmse123, predicteddanomaly12, predictedanomaly13, predictedanomaly123, var_total, var12, var13, var23, precision12, precision13, precision23, precision123, recall12, recall13, recall23, recall123, Accuracy12, Accuracy13, Accuracy23, Accuracy123, F112, F113, F123, F1123
    dict_to_return = {'predicteddanomaly123': predicteddanomaly123, 'stacking_values12': stacking_values12,
                      'stacking_values13': stacking_values13, 'stacking_values23': stacking_values23,
                      'stacking_values123': stacking_values123, 'rmse12': rmse12, 'rmse13': rmse13, 'rmse23': rmse23,
                      'rmse123': rmse123, 'predicteddanomaly12': predicteddanomaly12,
                      'predictedanomaly13': predictedanomaly13,
                      'predictedanomaly123': predictedanomaly123, 'var_total': var_total, 'var12': var12,
                      'var13': var13,
                      'var23': var23, 'precision12': precision12, 'precision13': precision13,
                      'precision23': precision23,
                      'precision123': precision123, 'recall12': recall12, 'recall13': recall13, 'recall23': recall23,
                      'recall123': recall123, 'Accuracy12': Accuracy12, 'Accuracy13': Accuracy13,
                      'Accuracy23': Accuracy23,
                      'Accuracy123': Accuracy123, 'F112': F112, 'F113': F113, 'F123': F123, 'F1123': F1123}
    return dict_to_return
