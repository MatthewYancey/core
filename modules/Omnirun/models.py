# a class that allows iterating through parameters and retuns the accuracy
import pandas as pd
import Levenshtein as lv
import tensorflow as tf
import time
from sklearn import neighbors, naive_bayes, tree
from model_analysis import cross_validation

class base_model_class:

    def __init__(self, tdm, cv_size, results_dir, model_name, observations = None, 
        seed = None, ratio = True, neighbors = None, max_depth = None, min_leaf_size = 1, nn_batch_size = None,
        nn_features_1 = None, nn_features_2 = None, nn_training_epochs = None, nn_learning_rate = None):
        self.data_tdm = tdm
        self.model_name = model_name
        self.cv_n = int(round(1 / cv_size))
        self.observations = observations
        self.seed = seed
        self.ratio = ratio
        self.neighbors = neighbors
        self.results_dir = results_dir
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.nn_batch_size = nn_batch_size
        self.nn_learning_rate = nn_learning_rate
        self.nn_training_epochs = nn_training_epochs
        self.nn_features_1 = nn_features_1
        self.nn_features_2 = nn_features_2

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

    # functions for preping data
    def factor_to_binary(self, y):
        # formats the y data set
        temp = pd.DataFrame(y)
        temp.columns = ['y']
        factor_set = list(set(y))
        for i in factor_set:
            temp.ix[:, str(i)] = 0
            temp.ix[y == i, str(i)] = 1

        temp = temp.drop('y', axis = 1)
        y = temp
        return(y, factor_set)

    def binary_to_factor(self, y, y_hat):
        pass

    # function for normalizing probabilities
    def norm_probs(self, probs):
        for i in range(len(probs)):
            probs[i] = probs[i] - min(probs[i])
            probs[i] = [p / sum(probs[i]) for p in probs[i]]
        return(probs)

class perceptron_nn_model(base_model_class):

    factor_set = []

    def run_model(self):
        # function for running the model
        def model_function(x_train, y_train, x_test, obs_x_train, obs_x_test):
            # preps some of the data
            x_train = x_train.values
            x_test = x_test.values
            res = self.factor_to_binary(y_train)
            y_train_tf = res[0].values
            factor_set = res[1]

            # Parameters
            learning_rate = self.nn_learning_rate
            training_epochs = int(self.nn_training_epochs)
            batch_size = int(self.nn_batch_size)
            display_step = 1

            # Network Parameters
            n_hidden_1 = int(self.nn_features_1) # 1st layer num features
            n_hidden_2 = int(self.nn_features_2) # 2nd layer num features
            n_input = x_train.shape[1] 
            n_classes = len(y_train_tf[0])

            # tf Graph input
            x = tf.placeholder("float", [None, n_input])
            y = tf.placeholder("float", [None, n_classes])

            # Create model
            def multilayer_perceptron(_X, _weights, _biases):
                layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
                layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
                return tf.matmul(layer_2, weights['out']) + biases['out']

            # Store layers weight & bias
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            }

            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }

            # Construct model
            model = multilayer_perceptron(x, weights, biases)

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y)) # Softmax loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

            # Initializing the variables
            init = tf.initialize_all_variables()

            with tf.Session() as sess:
                sess.run(init)

                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batches = int(x_train.shape[0] / batch_size)

                    # loops through all the batches
                    for i in range(total_batches):
                        start_i = i * batch_size
                        stop_i = start_i + batch_size
                        batch_xs = x_train[start_i:stop_i]
                        batch_ys = y_train_tf[start_i:stop_i]
                        # Fit training using batch data
                        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                        # Compute average loss
                        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batches

                    # Display logs per epoch step
                    if epoch % display_step == 0:
                        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

                # Test model
                correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
                
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print "Accuracy:", accuracy.eval({x: x_train, y: y_train_tf})
                
                # gets y_hat
                y_hat = tf.argmax(model, 1)
                y_hat = y_hat.eval(feed_dict={x: x_test})

                # gets the probabilities
                probs = model
                probs = probs.eval(feed_dict={x: x_test})

                probs = self.norm_probs(probs)

            # from the probabilities we get the y_hat values, this is faster than running the model again
            # take the y_hat values and translats them into the format they need to 
            y_hat_2 = []
            for i in y_hat:
                y_hat_2.append(factor_set[i])

            maxes = [max(p) for p in probs]
            # classes = sorted(y_train.unique())
            # indicies = [list(probs[i]).index(maxes[i]) for i in range(len(maxes))]
            # y_hat = [classes[i] for i in indicies]

            return([y_hat_2, maxes])

        # creates the cross validation object and passes in the functions it needs
        self.cv = cross_validation(self.data_tdm, model_function, self.cv_n, self.seed)
        # runs the cross validation
        cv_results = self.cv.run_cv()
        print('tensorflow perceptron model cross validation completed')

        # saves the results
        self.save_results('/' + self.model_name)

class decision_tree_model(base_model_class):

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
        
        # # creates the save name sufex
        # sufex = ''
        # if pd.notnull(self.max_depth):
        #     sufex + 'depth_' + str(int(self.max_depth)) + '_'
        # print(self.min_leaf_size)
        # sufex = sufex + 'leaf_' + str(int(self.min_leaf_size))

        # saves the results
        self.save_results('/' + self.model_name)

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
        self.save_results('/' + self.model_name)

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
                print('test hit')

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

        # saves the results
        self.save_results('/' + self.model_name)

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
        self.save_results('/' + self.model_name)