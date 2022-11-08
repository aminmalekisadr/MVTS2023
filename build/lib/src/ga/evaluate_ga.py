#from src.model.model import *
#from src.evaluation.evaluation import *
from src.data_handeling.Preprocess import Preprocess
#from src.ensemble.ensemble import *
from src.model.model import generate_rnn
from src.model.model import generate_rf
from src.model.model import generate_LinearRegression
from src.evaluation.evaluation import evaluate_rnn
from src.evaluation.evaluation import evaluate_rf
from src.evaluation.evaluation import evaluate_lr
import sys
import logging
import gc # garbage collector
import random
import yaml
import time
with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
# Load the data
    config_path= experiment_config["experiment_settings"]["config_path"]
collect_g=Preprocess(config_path)
collect_gc=collect_g.collect_gc()
mutation_rate = experiment_config["ga_parameters"]["mutation_rate"]
min_mutation_momentum= experiment_config["ga_parameters"]["min_mutation_momentum"]
max_mutation_momentum= experiment_config["ga_parameters"]["min_mutation_momentum"]
min_population= experiment_config["ga_parameters"]["min_population"]
max_population= experiment_config["ga_parameters"]["max_population"]
num_Iterations= experiment_config["ga_parameters"]["num_Iterations"]
optimisers= experiment_config["ml_parameters"]["optimizer_type"]
def crossover_rnn(model_1, model_2):
    """
    Executes crossover for the RNN in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    # Lower RMSE score has higher prob
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - (model_1.test_score / test_score_total)
    model_2_prob = 1 - model_1_prob
    # Probabilities of each item for each model (all items have same probabilities)
    model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
    model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

    # Number of layers of new generation depend on probability of each model
    num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

    # Create list of int with positions of the layers of both models.
    cross_layers_pos = []
    # Create list of weights
    weights = []
    # Add positions of layers for model 1. Input and ouput layer are not added.
    for i in range(2, len(model_1.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 1
        cross_layers_pos.append(mod_item)
        weights.append(model_1_prob_item)

    # Add positions of layers for model 2. Input and ouput layer are not added.
    for i in range(2, len(model_2.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 2
        cross_layers_pos.append(mod_item)
        weights.append(model_2_prob_item)

    collect_gc

    # If new num of layers are larger than the num crossover layers, keep num of crossover layers
    if num_layers_new_gen > len(cross_layers_pos):
        num_layers_new_gen = len(cross_layers_pos)

    # Randomly choose num_layers_new_gen layers of the new list
    cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False, p=weights))

    # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
    cross_layers = []
    for i in range(len(cross_layers_pos)):
        mod_item = cross_layers_pos[i]
        if mod_item.model == 1:
            cross_layers.append(model_1.layers[mod_item.pos])
        else:
            cross_layers.append(model_2.layers[mod_item.pos])

    collect_gc

    # Add input layer randomly from parent 1 or parent 2
    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.insert(0, model_1.layers[0])
    else:
        cross_layers.insert(0, model_2.layers[0])

    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.append(model_1.layers[len(model_1.layers) - 1])
    else:
        cross_layers.append(model_2.layers[len(model_2.layers) - 1])

    # Set new layers
    new_model._layers = cross_layers

    return new_model

def mutate_rnn(model):
    """
    Mutates the RNN model
    :param model:
    :return:
    """
    for i in range(len(model.layers)):
        # Mutate randomly each layer
        bit_random = random.uniform(0, 1)

        if bit_random <= mutation_rate:
            weights = model.layers[i].get_weights()  # list of weights as numpy arrays
            # calculate mutation momentum
            mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
            new_weights = [x * mutation_momentum for x in weights]
            model.layers[i].set_weights(new_weights)

    collect_gc


def crossover_rf(model_1, model_2):
    """
    Executes crossover for the RF in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - model_1.test_score / test_score_total
    model_2_prob = 1 - model_1_prob

    # New estimator is the sum of both estimators times their probability
    new_model.n_estimators = math.ceil(model_1.n_estimators * model_1_prob + model_2.n_estimators * model_2_prob)

    return new_model


def mutate_rf(model):
    """
    Mutates the Random Forest
    :param model:
    :return:
    """
    # Mutate randomly the estimator
    bit_random = random.uniform(0, 1)

    if bit_random <= mutation_rate:
        # calculate mutation momentum
        mutation_momentum = random.uniform(min_mutation_momentum, max_mutation_momentum)
        # Mutate estimators
        model.n_estimators = model.n_estimators + math.ceil(model.n_estimators * mutation_momentum)

    return model
def evaluate_ga(dataset, label_data):
    """
    Evaluates and generates the ensemble model using Genetic Algorithms
    :param dataset:
    :return:
    """
    print('#-----------------------------------------------')
    print('  ', dataset)
    print('#-----------------------------------------------')

    dataset1 = dataset
    process=Preprocess(config_path=config_path)

    dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st = process.load_dataset(
        dataset)

    start = time.time()  # Start Timer

    num_population = random.randint(experiment_config["ga_parameters"]["min_population"], experiment_config["ga_parameters"]["max_population"])  # Number of RNN to evaluate
    # == 1) Generate initial population for RNN and Random Forest
    population_rnn = []
    population_rf = []
    population_lr = []
    start_ga_1 = time.time()  # Start Timer

    for i in range(num_population):
        # -- RNN
        # Generate random topology configuration
        num_layers = random.randint(experiment_config["ml_parameters"]["min_num_layers"], experiment_config["ml_parameters"]["max_num_layers"])
        hidden_layers = []
        for j in range(num_layers):
            num_neurons = random.randint(experiment_config["ml_parameters"]["min_num_neurons"] , experiment_config["ml_parameters"]["max_num_neurons"] )
            hidden_layers.append(num_neurons)

        # collect_gc()

        # Generate and add rnn model to population
        model_rnn = generate_rnn(hidden_layers)
        print(model_rnn.summary())

        population_rnn.append(model_rnn)

        # -- RF
        # Generate random number of estimators for RF
        num_estimators = random.randint(experiment_config['ml_parameters']['min_num_estimators'], experiment_config['ml_parameters']['max_num_estimators'])

        # Generate and add rf model to population
        model_rf = generate_rf(num_estimators)
        population_rf.append(model_rf)
        population_lr.append(generate_LinearRegression())

    end_ga_1 = time.time() - start_ga_1  # End Timer
    print('Generate Initial population Time_Taken:%.3f' % end_ga_1)

    Preprocess(sys.argv[1]).collect_gc()
    # print(len(population))

    best_F1_rnn = 0.0
    best_F1_rf = 0.0
    best_F1_lr = 0.0
    best_rnn_model = None
    best_test_predict_rnn = None
    best_rf_model = None
    best_test_predict_rf = None
    # Evaluate fitness for
    for i in range(experiment_config['ga_parameters']['num_Iterations']):
        print('=================================================================================================')
        print(' iteration: %d, total iterations: %d, population size: %d ' % (i + 1, experiment_config['ga_parameters']['num_Iterations'], num_population))
        print('=================================================================================================')
        # train_score, test_score = float("inf"), float("inf")
        # == 2)  Evaluate fitness for population
        start_ga_2 = time.time()  # Start Timer
        for j in range(num_population):
            # Evaluate fitness for RNN
            rnn_model = population_rnn[j]
            train_MC_score_rnn, test_MC_score_rnn, train_score_rnn, test_score_rnn, train_predict_rnn, test_predict_rnn, \
            test_MC_dist_predict_rnn, test_MC_predict_point_rnn, train_MC_dist_predict_rnn, train_MC_predict_point_rnn, test_MC_predict_var_rnn, \
            train_MC_predict_var_rnn, anomalies_rnn, precision_rnn, recall_rnn, Accuracy_rnn, F1_rnn = evaluate_rnn(
                rnn_model, train_x_stf,
                test_x_stf, train_y,
                test_y, scaler,
                optimisers[0], dataset1[7:-4], label_data)
            # pdb.set_trace()

            try:
                if (F1_rnn > best_F1_rnn) and (math.isnan(F1_rnn) == False) and (math.isinf(F1_rnn) == False) and (
                        F1_rnn is not None):
                    best_rmse_rnn = test_score_rnn
                    # best_rnn_model = copy.copy(rnn_model)
                    best_rnn_model = rnn_model
                    best_test_predict_rnn = test_predict_rnn
                    best_train_MC_score_rnn = train_MC_score_rnn
                    best_test_MC_score_rnn = test_MC_score_rnn
                    best_train_score_rnn = train_score_rnn
                    best_test_score_rnn = test_score_rnn
                    best_train_predict_rnn = train_predict_rnn
                    best_test_MC_dist_predict_rnn = test_MC_dist_predict_rnn
                    best_test_MC_predict_point_rnn = test_MC_predict_point_rnn
                    best_train_MC_dist_predict_rnn = train_MC_dist_predict_rnn
                    best_train_MC_predict_point_rnn = train_MC_predict_point_rnn
                    best_test_MC_predict_var_rnn = test_MC_predict_var_rnn
                    best_train_MC_predict_var_rnn = train_MC_predict_var_rnn
                    best_anomalies_rnn = anomalies_rnn
                    best_F1_rnn = F1_rnn
                    best_Accuracy_rnn = Accuracy_rnn
                    best_recall_rnn = recall_rnn
                    best_precision_rnn = precision_rnn
      #              print('test predictions RNN: ', best_test_predict_rnn)
       #             print('test_score RMSE RNN:%.3f ' % best_test_score_rnn)
        #            print('test_score RMSE RNN_MC:%.3f ' % best_test_MC_score_rnn)
                    print('F1 RNN:%.3f ' % best_F1_rnn)
                    print('precision RNN_MC:%.3f ' % best_precision_rnn)
                    print('Recall RNN:%.3f ' % best_recall_rnn)
                    print('Accuracy RNN:%.3f ' % best_Accuracy_rnn)
                else:
                    F1_rnn = 0

            except:
                F1_rnn = 0
                pass
            # Evaluate fitness for RF

            rf_model = population_rf[j]

            train_score_rf, test_score_rf, train_predict_rf, test_predict_rf, variance_rf, anomalies_rf, precision_rf, \
            recall_rf, Accuracy_rf, F1_rf = evaluate_rf(
                rf_model, train_x_rf_st,
                test_x_rf_st, train_y_rf, test_y_rf,
                scaler, label_data)
            #pdb.set_trace()
            try:
                # print('test predictions RF: ', test_predict_rf)
                #  print('test_score RMSE RF:%.3f ' % test_score_rf)
                # print('F1 RF:%.3f ' % F1_rf)

                if (F1_rf > best_F1_rf) and (math.isnan(F1_rf) == False) and (math.isinf(F1_rf) == False) and (
                        F1_rf is not None):
                    best_rmse_rf = test_score_rf
                    # best_rf_model = copy.copy(rf_model)
                    best_rf_model = rf_model
                    best_test_predict_rf = rf_model.test_predict
                    best_train_score_rf = rf_model.train_score
                    best_test_score_rf = rf_model.test_score
                    best_train_predict_rf = rf_model.train_predict
                    best_test_predict_rf = rf_model.test_predict
                    best_variance_rf = rf_model.variance
                    best_anomalies_rf = rf_model.anomalies
                    best_F1_rf = F1_rf
                    best_Accuracy_rf = Accuracy_rf
                    best_recall_rf = recall_rf
                    best_precision_rf = precision_rf
     #               print('test predictions Rf: ', best_test_predict_rf)
     #               print('test_score RMSE Rf:%.3f ' % best_test_score_rf)
                    print('F1 Rf:%.3f ' % best_F1_rf)
                    print('precision Rf:%.3f ' % best_precision_rf)
                    print('Recall Rf:%.3f ' % best_recall_rf)
                    print('Accuracy Rf:%.3f ' % best_Accuracy_rf)

                else:
                    F1_rf = 0
            except:
                F1_rf = 0
                pass

        end_ga_2 = time.time() - start_ga_2  # End Timer
        print('Evaluate Fitness population Time_Taken:%.3f' % end_ga_2)

        # collect_gc()
        # == 3)  Select best individuals for next generation
        # pdb.set_trace()
        # Evaluate fitness for Leanier Regression

        lr_model = population_lr[j]
        train_predict_lr, test_predict_lr, variance_lr, anomalies_lr, precision_lr, \
        recall_lr, Accuracy_lr, F1_lr = evaluate_lr(
            lr_model, train_x_rf_st,
            test_x_rf_st, train_y_rf, test_y_rf,
            scaler, label_data)
        try:
         #    print('test predictions LR: ', test_predict_lr)
          #   print('test_score RMSE LR:%.3f ' % test_score_lr)
          print('F1 LR:%.3f ' % F1_lr)
            # pdb.set_trace()
          if (F1_lr > best_F1_lr) and (math.isnan(F1_lr) == False) and (math.isinf(F1_lr) == False) and (
                    F1_lr is not None):
               # best_rmse_lr = test_score_lr
                best_lr_model = copy.copy(lr_model)
                best_lr_model = lr_model
                best_test_predict_lr = lr_model.test_predict
                best_train_score_lr = lr_model.train_score
                best_test_score_lr = lr_model.test_score
                best_train_predict_lr = lr_model.train_predict
                best_test_predict_lr = lr_model.test_predict
                best_variance_lr = lr_model.variance
                best_anomalies_lr = lr_model.anomalies
                best_F1_lr = F1_lr
                best_Accuracy_lr = Accuracy_lr
                best_recall_lr = recall_lr
                best_precision_lr = precision_lr
               # print('test predictions LR: ', best_test_predict_lr)
               # print('test_score RMSE LR:%.3f ' % best_test_score_lr)
                print('F1 LR:%.3f ' % best_F1_lr)
                print('precision LR:%.3f ' % best_precision_lr)
                print('Recall LR:%.3f ' % best_recall_lr)
                print('Accuracy LR:%.3f ' % best_Accuracy_lr)
          else:
                F1_lr = 0
        except:
            F1_lr = 0
            pass

        # == 3) Create new population with new generations
        # Every generation will use the current best RNN and best RF to mate
        start_ga_3 = time.time()  # Start Timer
        for pop_index in range(num_population):
            # Select parents for mating
            # Element at pop_index as parent. This will be replaced with the new generation
            rnn_model_1 = population_rnn[pop_index]
            rf_model_1 = population_rf[pop_index]

            # 2 parent is the best found so far
            rnn_model_2 = best_rnn_model
            rf_model_2 = best_rf_model

            # == 4) Create new generation with crossover
            new_rnn_model = crossover_rnn(rnn_model_1, rnn_model_2)
            new_rf_model = crossover_rf(rf_model_1, rf_model_2)

            # == 5) Mutate new generation
            mutate_rnn(new_rnn_model)
            mutate_rf(new_rf_model)

            # Replace current model in population
            population_rnn[pop_index] = new_rnn_model
            population_rf[pop_index] = new_rf_model

        end_ga_3 = time.time() - start_ga_3  # End Timer
        print('Generate new population Time_Taken:%.3f' % end_ga_3)

        collect_gc

    collect_gc

    end = time.time() - start  # End Timer
    # pdb.set_trace()

    print('=============== BEST RNN ===============')
    # print('Best predictions: ', [x[0] for x in best_test_predict_rnn])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rnn, end))
    # save_plot_model_rnn(best_rnn_model)

    print('=============== BEST RF ===============')
    # print('Best predictions: ', [x for x in best_test_predict_rf])
    print('Best RMSE:%.3f Time_Taken:%.3f' % (best_rmse_rf, end))

    # save_plot_model_rf(best_rf_model)
    # print(best_rf_model.get_params(deep=True))

    # Ensemble
    print('=============== Ensemble ===============')
    # pdb.set_trace()

    averaging_values, stacking_values_uncertainty, rmse1, rmse2, anomalies_merged, var_merge, var_total_uncertainty = ensemble_stacking(
        best_rnn_model, best_rf_model, best_test_predict_rnn, best_rmse_rnn, best_test_predict_rf, best_rmse_rf, test_y,
        scaler, dataset1[7:-4])
    # print('Ensemble averaging_values: ', averaging_values)
    #  print('Ensemble uncertainty_values: ', stacking_values_uncertainty)
    print('Ensemble rmse: ', rmse1)
    print('Ensemble rmse uncertainty: ', rmse2)
    return best_anomalies_rf, best_anomalies_rnn, anomalies_merged, var_merge, best_variance_rf, best_test_MC_predict_var_rnn, best_F1_rnn, best_F1_rf, best_Accuracy_rnn, best_Accuracy_rf, best_precision_rnn, best_precision_rf, best_recall_rnn, best_recall_rf

