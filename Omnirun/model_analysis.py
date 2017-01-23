import pandas as pd
import random
import time

class cross_validation:

    def __init__(self, data_tdm, modeling_function, cv, set_seed = None, observations = None):
        self.modeling_function = modeling_function
        self.cv = cv
        self.observations = observations

        # strips out the y value from the tdm
        data_y = data_tdm['_y_'].apply(str)
        data_y.columns = ['y']
        self.data_y = data_y
        data_tdm = data_tdm.drop('_y_', axis = 1)
        self.data_tdm = data_tdm

        # need to figur out how to fix this        
        # # sets the seed if desired
        # print(set_seed)
        # if set_seed is not None:    
        #     random.seed( set_seed )

    def run_cv(self):
        time_all_s = time.clock()
        time_modeling = 0

        # creates an index for spliting the data into testing and training
        test_index = [random.randint(1, self.cv) for i in self.data_tdm.index]

        # for storing the results of each cv iteration
        self.data_y_hat = pd.DataFrame(self.data_y)
        self.data_y_hat['y_hat'] = ''
        self.data_y_hat['y_hat_prob'] = ''
        self.data_y_hat.columns = ['y', 'y_hat', 'y_hat_prob']

        # iterates through each cv segment
        for this_cv in range(1, self.cv + 1):

            # subsets the data to test and train
            data_tdm_train = self.data_tdm.ix[[t != this_cv for t in test_index], :]
            data_y_train = self.data_y.ix[[t != this_cv for t in test_index]]
            data_tdm_test = self.data_tdm.ix[[t == this_cv for t in test_index], :]

            if self.observations is not None:
                obs_train = self.observations[[t != this_cv for t in test_index]]
                obs_test = self.observations[[t == this_cv for t in test_index]]
            else:
                obs_train = None
                obs_test = None
            
            # sends the data to the modeling function
            time_model_s = time.clock()            
            model_results = self.modeling_function(data_tdm_train, data_y_train, data_tdm_test, obs_train, obs_test)

            # places the result in the correct rows
            # probabilities = self.prob_function(data_tdm_train, data_y_train, data_tdm_test)
            # maxes = [max(p) for p in probabilities]
            # y_hat = self.y_hat_function(probabilities, data_y_train)
            time_modeling += time.clock() - time_model_s

            self.data_y_hat.ix[[t == this_cv for t in test_index], 'y_hat'] = model_results[0]
            self.data_y_hat.ix[[t == this_cv for t in test_index], 'y_hat_prob'] = model_results[1]

            print('Fold ' + str(this_cv) + ' completed')

        # calculates the accuracy
        accuracy = sum(self.data_y_hat.y == self.data_y_hat.y_hat) / float(self.data_y_hat.shape[0])

        print('Modeling Time: ' + str(time_modeling))
        print('Total Time: ' + str(time.clock() - time_all_s))
        
        return([accuracy])