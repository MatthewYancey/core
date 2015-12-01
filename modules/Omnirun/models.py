# a class that allows iterating through parameters and retuns the accuracy
from sklearn import neighbors, naive_bayes, tree
import pandas as pd
import Levenshtein as lv
from model_analysis import cross_validation
import time


class base_model_class:

    def __init__(self, tdm, cv_size, results_dir, observations = None, 
        seed = None, ratio = True, neighbors = None, max_depth = None, min_leaf_size = 1):
        self.data_tdm = tdm
        self.cv_n = int(round(1 / cv_size))
        self.observations = observations
        self.seed = seed
        self.ratio = ratio
        self.neighbors = neighbors
        self.results_dir = results_dir
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

        # starts the timer and runs the model
        time_s = time.clock()
        self.run_model()

    def save_results(self, file_name):
        
        # gets the estimated values from the model analysis object
        self.results = self.cv.data_y_hat
        
        # adds the observations if they were passed in            
        if self.observations is not None:
            self.results['observations'] = self.observations

        # creates the name for the saved file
        results_path = self.results_dir + file_name + '.csv'
        
        # saves the data
        self.results.to_csv(results_path, index = False)


class tree_model(base_model_class):

    def run_model(self):
        # function for running the model
        def model_function(x_train, y_train, x_test, obs_x_train, obs_x_test):

            # creates the knn model and gets the probabilities
            decision_tree = tree.DecisionTreeClassifier()
            decision_tree.fit(x_train, y_train)
            probs = decision_tree.predict_proba(x_test)

            # from the probabilities we get the y_hat values, this is faster than running the model again
            maxes = [max(p) for p in probs]
            classes = sorted(y_train.unique())
            indicies = [list(probs[i]).index(maxes[i]) for i in range(len(maxes))]
            y_hat = [classes[i] for i in indicies]
            return([y_hat, maxes])

        # creates the cross validation object and passes in the functions it needs
        self.cv = cross_validation(self.data_tdm, model_function, self.cv_n, self.seed, self.observations)

        # runs the cross validation
        cv_results = self.cv.run_cv()
        print('Decision Tree model cross validation completed')
        
        # creates the save name sufex
        sufex = ''
        if pd.notnull(self.max_depth):
            sufex + 'depth_' + str(int(self.max_depth)) + '_'
        sufex = sufex + 'leaf_' + str(int(self.min_leaf_size))

        # saves the results
        self.save_results('/tree_results_' + sufex)

        # returns the summary stats to be appended
        return(cv_results)

class knn_model(base_model_class):

    def run_model(self):        
        # function for running the model
        def model_function(x_train, y_train, x_test, obs_x_train, obs_x_test):

            # creates the knn model and gets the probabilities
            knn = neighbors.KNeighborsClassifier(n_neighbors = int(self.neighbors))
            knn.fit(x_train, y_train)
            probs = knn.predict_proba(x_test)

            # from the probabilities we get the y_hat values, this is faster than running the model again
            maxes = [max(p) for p in probs]
            classes = sorted(y_train.unique())
            indicies = [list(probs[i]).index(maxes[i]) for i in range(len(maxes))]
            y_hat = [classes[i] for i in indicies]
            return([y_hat, maxes])

        # creates the cross validation object and passes in the functions it needs
        self.cv = cross_validation(self.data_tdm, model_function, self.cv_n, self.seed, self.observations)

        # runs the cross validation
        cv_results = self.cv.run_cv()
        print('KNN model cross validation completed')
        # saves the results
        self.save_results('/knn_results_n' + str(int(self.neighbors)))

        # returns the summary stats to be appended
        return(cv_results)


class levenshtein_model(base_model_class):

    def run_model(self):        
        # function for running the model
        def model_function(x_train, y_train, x_test, obs_x_train, obs_x_test):
            y_hat = []
            maxes = []
            y_train_list = y_train.tolist()

            for test in obs_x_test:

                if self.ratio:
                    distances = [lv.ratio(test, train) for train in obs_x_train]
                    best_distance = max(distances)
                else:
                    distances = [lv.distance(test, train) for train in obs_x_train]
                    best_distance = min(distances)

                # finds the best match and appends the results to the res list
                match_index = distances.index(best_distance)
                y_hat.append(y_train_list[match_index])
                maxes.append(best_distance)

            # returns the results for the cv object
            return(y_hat, maxes)

        # creates the cross validation object and passes in the function it needs
        self.cv = cross_validation(self.data_tdm, model_function, self.cv_n, self.seed, self.observations)
        # runs the cross validation
        cv_results = self.cv.run_cv()
        print('Levenshtein model cross validation completed')

        # saves the results but first edits the path string to get just the directory
        if self.ratio:
            self.save_results('/lv_results_ratio')
        else:
            self.save_results('/lv_results_distance')


class naive_bayes_model(base_model_class):

    def run_model(self):        
        # function for running the model
        def model_function(x_train, y_train, x_test, obs_x_train, obs_x_test):

            # creates the knn model and gets the probabilities
            nb = naive_bayes.GaussianNB()
            nb.fit(x_train, y_train)
            probs = nb.predict_proba(x_test)

            # from the probabilities we get the y_hat values, this is faster than running the model again
            maxes = [max(p) for p in probs]
            classes = sorted(y_train.unique())
            indicies = [list(probs[i]).index(maxes[i]) for i in range(len(maxes))]
            y_hat = [classes[i] for i in indicies]
            return([y_hat, maxes])

        # creates the cross validation object and passes in the functions it needs
        self.cv = cross_validation(self.data_tdm, model_function, self.cv_n, self.seed)
        # runs the cross validation
        cv_results = self.cv.run_cv()
        print('Naive Bayes model cross validation completed')

        # saves the results
        self.save_results('/nb_results')