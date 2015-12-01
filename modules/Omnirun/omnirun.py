import models
import pandas as pd
import random as rn

# class for handeling classification
class classification(object):
    
    def __init__(self, parameters_file_path):

        results_summary = []

        # creates a dictionary with the models
        model_dict = {'knn': models.knn_model, 
            'lev': models.levenshtein_model,
            'naive bayes': models.naive_bayes_model, 
            'tree': models.tree_model}

        # reads in the parameters files
        parameters = pd.read_csv(parameters_file_path)

        # reads in the data
        data = pd.read_csv(parameters.ix[0, 'input_data'])
        print('Data imported')
        
        # sets the seed if we need to
        if pd.notnull(parameters.ix[0, 'seed']):
            print('Setting seed')
            rn.seed = parameters.ix[0, 'seed']

        # subsets the data
        if pd.notnull(parameters.ix[0, 'sample_size']):
            data = data.ix[rn.sample(data.index, parameters.ix[0, 'sample_size']), :]
            print('Data subsetted')

        # seperates out the raw text if we have it
        if 'x_raw' in data.columns:
            observations = data['x_raw']
            data = data.drop('x_raw', 1)
            # column_limit includes or excludes the observations in the summary tables
            column_limit = 3
        else:
            observations = None
            column_limit = 2

        # goes through each of the models specified in the parameters csv
        for i, row in parameters.iterrows():
            model = model_dict[row['model'].lower()](data, row['cv_holdout'], row['save_path'], observations, 
                                                          row['seed'], row['ratio'], row['neighbors'], 
                                                          row['tree_depth'], row['min_leaf_size'])
            model_results = model.results

            # generates some accuracy stats that will be used in the results_summary data frame
            accuracy = sum(model_results['y'] == model_results['y_hat']) / float(model_results.shape[0])
            results_summary.append([row['model_name'], accuracy])

            # takes the results and appends them to a data frame
            if 'results' in locals():
                # ADD IN A TEST TO SEE IF THE OBSERVATION COLUMNS MATCH
                # drops variables I already have
                if observations is None:
                    model_results = model_results.drop(['y'], axis = 1)
                else:
                    model_results = model_results.drop(['y', 'observations'], axis = 1)
                # makes the column names unique so we can append all the data together
                model_results.columns = [c + '_' + row['model_name'] for c in model_results.columns]
                # adds the new columns to the results
                results = pd.concat([results, model_results], axis = 1)
            else:
                results = model_results
                # makes some of the column names unique
                print(model_results.columns)
                new_column_names = [c + '_' + row['model_name'] for c in model_results.columns]
                if observations is not None:
                    new_column_names = [model_results.columns[0]] + new_column_names[1:3] + [model_results.columns[3]]
                else:
                    new_column_names = [model_results.columns[0]] + new_column_names[1:3]
                model_results.columns = new_column_names

        # saves the results
        print('saved results')
        results.to_csv(parameters.ix[0, 'save_path'] + '/results_all.csv', index = False)

        # does some formatting and saves the results again
        original_columns = list(results.columns.values)

        # modifies the columns names
        keep_list = ['y', 'observations']
        columns = original_columns[:]
        for i in range(len(columns)):
            if 'y_hat' in columns[i] and 'y_hat_prob' not in columns[i]:
                columns[i] = columns[i].replace('y_hat_', '')
                keep_list.append(columns[i])

        # makes a copy of the results data and keeps only specific columns
        data_01 = results
        data_01.columns = columns
        data_01 = data_01.ix[:, keep_list]
        # melts the data
        data_01 = pd.melt(data_01, id_vars = ['y', 'observations'], var_name = 'model', value_name = 'y_hat')
        
        # does the same thing as above but for the probs
        keep_list = ['y', 'observations']
        columns = original_columns[:]
        for i in range(len(columns)):
            if 'y_hat_prob' in columns[i]:
                columns[i] = columns[i].replace('y_hat_prob_', '')
                keep_list.append(columns[i])
        data02 = results
        data02.columns = columns
        data02 = data02.ix[:, keep_list]
        data02 = pd.melt(data02, id_vars = ['y', 'observations'], var_name = 'model', value_name = 'y_hat_prob')

        # combines the two results
        data_01['y_hat_prob'] = data02.y_hat_prob
        # saves the newly modified results
        data_01.to_csv(parameters.ix[0, 'save_path'] + '/results_all_long.csv', index = False)

        # creates a summary results dataframe
        results_summary = pd.DataFrame(results_summary)
        results_summary.columns = ['model_name', 'accuracy']
        results_summary.to_csv(parameters.ix[0, 'save_path'] + '/results_summary.csv', index = False)