# does all the training
import os
import sys
import glob
import dlib
# from skimage import io

path = os.getcwd() + '/../'

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
# The trainer is a kind of support vector machine and therefore has the usual
# SVM C parameter.  In general, a bigger C encourages it to fit the training
# data better but might lead to overfitting.  You must find the best C value
# empirically by checking how well the trained detector works on a test set of
# images you haven't trained on.  Don't just leave the value set at 5.  Try a
# few different C values and see what works best for your data.
options.C = 5
# Tell the code how many CPU cores your computer has for the fastest training.
options.num_threads = 2
options.be_verbose = True

training_xml_path = os.path.join(path, 'data/mdl_detector/train.xml')
testing_xml_path = os.path.join(path, 'data/mdl_detector/test.xml')

# does the training
dlib.train_simple_object_detector(training_xml_path, path + 'data/detector.svm', options)
