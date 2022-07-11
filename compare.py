# Adversarial Learning
# Creates and applies binary input detector model for adversarial image detection
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (04/14/20)
# Updated (MM/DD/YY)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
from art.defences.trainer import AdversarialTrainerMadryPGD
from art.defences.trainer import AdversarialTrainerFBF
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import random
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, ProjectedGradientDescent
from art.data_generators import KerasDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import load_dataset
from attacks import create_attack
from detector.configs import path_fig, path, max_iter, num_random_init
from utils import get_acc_preds, plot_attacks_acc
from art.defences.detector.evasion import BinaryInputDetector
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from models.detector_models import resnet_model, cnn_model, densenet_model, alexnet_model, vgg19_model
from art.utils import get_file
from art import config
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# query user for dataset
dataset = input('Enter dataset to be used (brain_mri, mnist, cifar, ddsm, lidc)\n')
model_type = input('Enter model type for detector model (resnet, densenet, svm, alexnet, cnn, randomforest)\n')


# load in dataset using load_dataset function
if dataset == 'lidc':
    aug = False
else:
    aug = True


x_train, y_train, x_test, y_test = load_dataset(dataset, path, aug=aug)


# query user for adversarial attack to use for generating adversarial test set
attack_type = input('Enter attack to be used (fgsm, pgd, bim, jsma)\n')
if (attack_type != 'fgsm') & (attack_type != 'pgd') & (attack_type != 'bim') & (attack_type != 'jsma'):
    print('attack type not supported\n')
    exit(0)


# verify that x_train, y_train, x_test, y_test have the correct dimensions
print('x_train shape: ', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

nb_train = x_train.shape[0]
nb_test = x_test.shape[0]


# load classification model
# model = load_model(path + 'models/' + dataset + '_vgg16_model.h5')
model = load_model(path + 'mys3bucket/' + 'models/' + dataset + '_vgg16_model.h5')


# load adversarially trained model
robust_model = load_model(path + 'models/' + dataset + '/' + dataset + '_vgg16_robust_classifier.h5')


# load input shape
if dataset == 'brain_mri':
    input_shape = [224, 224, 3]
elif dataset == 'mnist':
    input_shape = [32, 32, 3]
elif dataset == 'ddsm':
    input_shape = [116, 116, 3]
elif dataset == 'cifar':
    input_shape = [32, 32, 3]
elif dataset == 'lidc':
    input_shape = (116, 116, 3)

# convert model to KerasClassifier
classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)
robust_classifier = KerasClassifier(clip_values=(0, 1), model=robust_model, use_logits=False)

# # Craft adversarial samples
attacker = create_attack(attack_type, classifier)

if dataset == 'cifar':
    epsilon = 0.05
elif dataset == 'lidc':
    epsilon = 0.006
elif dataset == 'brain_mri':
    # epsilon = 0.004
    epsilon = 0.008
    # epsilon = 0.006
elif dataset == 'ddsm':
    epsilon = 0.004 # official
    # epsilon = 0.006 # just try for logistic reg
elif dataset == 'mnist':
    epsilon = 0.1


try:
    attacker.set_params(**{'eps_step': epsilon / 4})
except:
    pass
attacker.set_params(**{'eps': epsilon})

# load detection models
if model_type == 'logisticreg' or model_type == 'svm2' or model_type == 'rsf2':
    import pickle
    pkl_filename = path + 'mys3bucket/detector/' + dataset + '/pickle_model_' + model_type + '_detector_' + attack_type + '_' + str(epsilon) + '.pkl'
    detector = pickle.load(open(pkl_filename, 'rb'))

else:
    detector = load_model(path + 'detector/' + dataset + '/' + model_type + '_detector_' + attack_type + '_' + str(epsilon) + '.h5')


epsilon = 0.004
# epsilon = 0.002
try:
    attacker.set_params(**{'eps_step': epsilon / 4})
except:
    pass
attacker.set_params(**{'eps': epsilon})


# create combined dataset: half normal images and half adversarial images
x_test_adv = attacker.generate(x_test)
x_test_detector = np.concatenate((x_test, x_test_adv), axis=0)
nb_test = x_test.shape[0]
y_test_detector = np.concatenate((np.array([[1,0]]*nb_test), np.array([[0,1]]*nb_test)), axis=0)
y_test_labels = np.concatenate((y_test, y_test), axis=0)

# show accuracy of original classifier on combined dataset
acc1 = accuracy_score(y_test_labels, (classifier.predict(x_test_detector)>0.5))
print("Accuracy of original classifier on combined test set: %.2f%%, " % (acc1 * 100))

# show accuracy of adversarially trained classifier on combined dataset
acc2 = accuracy_score(y_test_labels, (robust_classifier.predict(x_test_detector)>0.5))
print("Accuracy of robust classifier on combined test set: %.2f%%, " % (acc2 * 100))

# create dataset of images not detected as adversarial
x_new = []
y_new = []
# remove images classified as adversarial by detector
for i in range(x_test_detector.shape[0]):
    img = np.expand_dims(x_test_detector[i], axis=0)
    # if image is classified as normal by detector, include in new dataset
    if ((detector.predict(img)>0.5) == [1, 0]).all():
        x_new.append(x_test_detector[i])
        y_new.append(y_test_labels[i])

nb_new_images = np.asarray(x_new).shape[0]
print("number of images remaining after exclusion: ", nb_new_images)

# show accuracy of original classifier on dataset after adversarial detection
acc3 = accuracy_score(y_new, (classifier.predict(x_new)>0.5))
# get number of images in new dataset correctly classified by classified
# this includes normal images correctly not picked up by detector and classified correctly by classifier
# as well as adversarial images incorrectly undetected by detector and classified correctly by classifier
nb_correct_classified_images = acc3 * nb_new_images


# get number of normal images incorrectly detected as adversarial
nb_normal_detect_adv = 0
for i in range(x_test_detector.shape[0]):
    # if image is normal
    if (y_test_detector[i] == [1, 0]).all():
        # if detector classifies image as adversarial
        img = np.expand_dims(x_test_detector[i], axis=0)
        if ((detector.predict(img) > 0.5) == [0, 1]).all():
            nb_normal_detect_adv = nb_normal_detect_adv + 1
# print(nb_normal_detect_adv)

acc4 = nb_correct_classified_images / (nb_new_images + nb_normal_detect_adv)
print("Accuracy of original classifier after using adversarial detection: %.2f%%, " % (acc4 * 100))

# show accuracy of robust classifier after adversarial detection
nb_correct_classified_images = accuracy_score(y_new, (robust_classifier.predict(x_new)>0.5)) * nb_new_images

acc5 = nb_correct_classified_images / (nb_new_images + nb_normal_detect_adv)
print("Accuracy of robust classifier after using adversarial detection: %.2f%%, " % (acc5 * 100))
