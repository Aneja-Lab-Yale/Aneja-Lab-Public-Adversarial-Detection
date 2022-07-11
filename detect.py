# Adversarial Learning
# Creates and applies binary input detector model for adversarial image detection
# Aneja Lab | Yale School of Medicine
# Marina Joel
# Created (11/13/20)
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
from detector_models import resnet_model, densenet_model
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


# load model
# model = load_model(path + 'models/' + dataset + '_vgg16_model.h5')
model = load_model(path + 'mys3bucket/' + 'models/' + dataset + '_vgg16_model.h5')

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

# evaluate classifier accuracy on training set
acc = accuracy_score(y_train, (classifier.predict(x_train)>0.5))
print("Accuracy of original classifier on training set: %.2f%%, " % (acc * 100))

acc = accuracy_score(y_test, (classifier.predict(x_test)>0.5))
print("Accuracy of original classifier on test set: %.2f%%, " % (acc * 100))


# evaluate original classifier accuracy on training set
# auc = metrics.roc_auc_score(y_train, (classifier.predict(x_train)>0.5))
# print("\nAUC of original classifier on training set: %.2f%%" % (auc * 100))
#
# # Evaluate the classifier on the test set
# auc = metrics.roc_auc_score(y_test, (classifier.predict(x_test)>0.5))
# print("\nAUC of original classifier on test set: %.2f%%" % (auc * 100))


# Craft adversarial samples
attacker = create_attack(attack_type, classifier)
x_test_adv = attacker.generate(x_test)
x_train_adv = attacker.generate(x_train)


# Evaluate the classifier on the adversarial test examples
acc = accuracy_score(y_train, (classifier.predict(x_train_adv)>0.5))
print("\nAccuracy of original classifier on adversarial train set: %.2f%%" % (acc * 100))


# Evaluate the classifier on the adversarial test examples
acc = accuracy_score(y_test, (classifier.predict(x_test_adv)>0.5))
print("\nAccuracy of original classifier on adversarial test set: %.2f%%" % (acc * 100))


# load detector model
if model_type == 'resnet':
    detector = resnet_model(input_shape)
elif model_type == 'densenet':
    detector = densenet_model(input_shape)


if dataset == 'cifar':
    epsilon = 0.05
elif dataset == 'lidc':
    epsilon = 0.006
elif dataset == 'brain_mri':
    epsilon = 0.008
elif dataset == 'ddsm':
    epsilon = 0.004 
elif dataset == 'mnist':
    epsilon = 0.1

try:
    attacker.set_params(**{'eps_step': epsilon / 4})
except:
    pass
attacker.set_params(**{'eps': epsilon})

x_train_adv = attacker.generate(x_train)
x_test_adv = attacker.generate(x_test)
x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
y_train_detector = np.concatenate((np.array([[1,0]]*nb_train), np.array([[0,1]]*nb_train)), axis=0)
x_test_detector = np.concatenate((x_test, x_test_adv), axis=0)
y_test_detector = np.concatenate((np.array([[1,0]]*nb_test), np.array([[0,1]]*nb_test)), axis=0)


np.save(path + 'mys3bucket/detector/' + dataset + '/x_train_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy', x_train_detector)
np.save(path + 'mys3bucket/detector/' + dataset + '/y_train_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy', y_train_detector)
np.save(path + 'mys3bucket/detector/' + dataset + '/x_test_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy', x_test_detector)
np.save(path + 'mys3bucket/detector/' + dataset + '/y_test_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy', y_test_detector)
np.save(path + 'mys3bucket/detector/' + dataset + '/y_test_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy', y_test_detector)

if dataset == 'lidc':
    x_train_detector = np.load(path + 'detector/' + dataset + '/x_train_detector_' + attack_type + '_' + str(epsilon) + '.npy' )
    y_train_detector = np.load(path + 'detector/' + dataset + '/y_train_detector_' + attack_type + '_' + str(epsilon) + '.npy' )
    x_test_detector = np.load(path + 'detector/' + dataset + '/x_test_detector_' + attack_type + '_' + str(epsilon)  + '.npy' )
    y_test_detector = np.load(path + 'detector/' + dataset + '/y_test_detector_' + attack_type + '_' + str(epsilon) + '.npy' )

else:
    x_train_detector = np.load(path + 'mys3bucket/detector/' + dataset + '/x_train_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy',allow_pickle=True )
    y_train_detector = np.load(path + 'mys3bucket/detector/' + dataset + '/y_train_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy',allow_pickle=True )
    x_test_detector = np.load(path + 'mys3bucket/detector/' + dataset + '/x_test_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy',allow_pickle=True )
    y_test_detector = np.load(path + 'mys3bucket/detector/' + dataset + '/y_test_detector_' + attack_type + '_' + str(epsilon) + 'aug_' + str(aug) + '.npy',allow_pickle=True )


# train model
if model_type == 'svm':
    from tensorflow.keras.applications import DenseNet121

    model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x_train_features = model.predict(x_train_detector, batch_size=64)
    print(x_train_features.shape)
    x_train_detector = x_train_features.reshape(
        (x_train_features.shape[0], x_train_features.shape[1] * x_train_features.shape[2] * x_train_features.shape[3]))

    x_test_features = model.predict(x_test_detector, batch_size=64)
    print(x_test_features.shape)
    x_test_detector = x_test_features.reshape(
        (x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))

    y_train_detector = np.argmax(y_train_detector, axis=1)
    y_test_detector = np.argmax(y_test_detector, axis=1)
    detector = LinearSVC(max_iter=5000)
    detector.fit(x_train_detector, y_train_detector)

elif model_type == 'rsf':
    from tensorflow.keras.applications import DenseNet121

    model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x_train_features = model.predict(x_train_detector, batch_size=64)
    print(x_train_features.shape)
    x_train_detector = x_train_features.reshape(
        (x_train_features.shape[0], x_train_features.shape[1] * x_train_features.shape[2] * x_train_features.shape[3]))

    x_test_features = model.predict(x_test_detector, batch_size=64)
    print(x_test_features.shape)
    x_test_detector = x_test_features.reshape(
        (x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))

    y_train_detector = np.argmax(y_train_detector, axis=1)
    y_test_detector = np.argmax(y_test_detector, axis=1)
    detector = RandomForestClassifier()
    detector.fit(x_train_detector, y_train_detector)
    
elif model_type == 'logisticreg':
    from sklearn.metrics import classification_report
    import numpy as np
    import pickle
    from tensorflow.keras.applications import DenseNet121
    model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x_train_features = model.predict(x_train_detector, batch_size=64)
    print(x_train_features.shape)
    x_train_detector = x_train_features.reshape((x_train_features.shape[0], x_train_features.shape[1] * x_train_features.shape[2] * x_train_features.shape[3]))

    x_test_features = model.predict(x_test_detector, batch_size=64)
    print(x_test_features.shape)
    x_test_detector = x_test_features.reshape((x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))

    y_train_detector = np.argmax(y_train_detector ,axis=1)
    y_test_detector = np.argmax(y_test_detector, axis=1)

    detector = LogisticRegression(solver="lbfgs", multi_class="auto",
                               max_iter=1000)
    detector.fit(x_train_detector, y_train_detector)



else:
    model_path = path + 'detector/' + dataset + '/' + model_type + '_detector_' + attack_type + '_' + str(epsilon) + '.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    history = detector.fit(x_train_detector, y_train_detector, batch_size=64, epochs=30, callbacks=[checkpoint], validation_data=(x_test_detector, y_test_detector), verbose=True)
    # detector = load_model(model_path)



if model_type == 'svm' or model_type == 'randomforest' or model_type == 'logisticreg':
    import pickle
    pkl_filename = path + 'mys3bucket/detector/' + dataset + '/pickle_model_' + model_type + '_detector_' + attack_type + '_' + str(epsilon) + '.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(detector, file)
    detector = pickle.load(open(pkl_filename, 'rb'))

else:
    detector = load_model(path + 'detector/' + dataset + '/' + model_type + '_detector_' + attack_type + '_' + str(epsilon) + '.h5')
# # Evaluate the detector on the combined training samples
# auc = metrics.roc_auc_score(y_train_detector, (detector.predict(x_train_detector)>0.5))
# print("\nAUC of detector on combined training set: %.2f%%" % (auc * 100))
#
# # Evaluate the detector on the combined testing samples
# auc = metrics.roc_auc_score(y_test_detector, (detector.predict(x_test_detector)>0.5))
# print("\nAUC of detector on combined test set: %.2f%%" % (auc * 100))

# # Evaluate the detector on the normal test examples
# y_test = np.array([[1,0]]*nb_test)
# if model_type == 'svm':
#     y_test = np.argmax(y_test, axis=1)
# acc = accuracy_score(y_test, (detector.predict(x_test)>0.5))
# print("\nAccuracy of detector on normal test set: %.2f%%" % (acc * 100))
#
# # Evaluate the detector on the adversarial test examples
# y_test_adv = np.array([[0,1]]*nb_test)
# if model_type == 'svm':
#     y_test = np.argmax(y_test, axis=1)
# if model_type == 'svm':
#     y_test_adv = np.argmax(y_test_adv, axis=1)
# acc = accuracy_score(y_test_adv, (detector.predict(x_test_adv)>0.5))
# print("\nAccuracy of detector on adversarial test set: %.2f%%" % (acc * 100))
#



# detector_auc_fgsm = []
# detector_auc_bim = []
# detector_auc_pgd = []

detector_acc_fgsm = []
detector_acc_bim = []
detector_acc_pgd = []
eps_range = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]


x_train, y_train_null, x_test, y_test_null = load_dataset(dataset, path, aug=True)


step_size = 0.001
fig, ax = plt.subplots()
ax.plot(np.array(eps_range) / step_size, np.array(detector_auc_fgsm) * 100, 'b--', label='FGSM')
ax.plot(np.array(eps_range) / step_size, np.array(detector_auc_pgd) * 100, 'r--', label='PGD')
ax.plot(np.array(eps_range) / step_size, np.array(detector_auc_bim) * 100, 'g--', label='BIM')



acc_fgsm = []
acc_pgd = []
acc_bim = []
for eps in eps_range:
    attacker_fgsm = FastGradientMethod(classifier, eps=eps)
    attacker_pgd = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps / 4, max_iter=10,
                                            num_random_init=5)
    attacker_bim = BasicIterativeMethod(classifier, eps=eps, eps_step=eps / 10, max_iter=10)
    x_test_fgsm = attacker_fgsm.generate(x_test)
    x_test_pgd = attacker_pgd.generate(x_test)
    x_test_bim = attacker_bim.generate(x_test)
    acc_fgsm += [metrics.accuracy_score(y_test, (classifier.predict(x_test_fgsm) > 0.5))]
    acc_pgd += [metrics.accuracy_score(y_test, (classifier.predict(x_test_pgd) > 0.5))]
    acc_bim += [metrics.accuracy_score(y_test, (classifier.predict(x_test_bim) > 0.5))]


    print('eps:', eps)
    # if model_type == 'svm':
    #     x_test_fgsm = x_test_fgsm.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    #     x_test_pgd = x_test_pgd.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    #     x_test_bim = x_test_bim.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])



    x_test_combined = np.concatenate((x_test, x_test_fgsm), axis=0)
    y_test_combined = np.concatenate((np.array([[1, 0]] * nb_test), np.array([[0, 1]] * nb_test)), axis=0)
    if model_type == 'svm' or model_type == 'randomforest':
        x_test_combined = x_test_combined.reshape(x_test_combined.shape[0], x_test_combined.shape[1] * x_test_combined.shape[2] * x_test_combined.shape[3])
        y_test_combined = np.argmax(y_test_combined, axis=1)


    elif model_type == 'logisticreg' or model_type == 'svm2' or model_type == 'rsf2':
        x_test_features = model.predict(x_test_combined, batch_size=64)
        x_test_combined = x_test_features.reshape((x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))
        y_test_combined = np.argmax(y_test_combined, axis=1)

    else:
        y_test_combined = np.argmax(y_test_combined, axis=1)
    print(y_test_combined[:10])
    print(np.argmax((detector.predict(x_test_combined) > 0.5), axis=1)[:10])
    acc1 = metrics.accuracy_score(y_test_combined, np.argmax((detector.predict(x_test_combined) > 0.5), axis = 1))
    detector_acc_fgsm += [acc1]
    print("ACC of detector on combined normal and fgsm adversarial test set: %.2f%%" % (100 * acc1))

    # Evaluate acc of detector of combined normal and adversarial samples
    x_test_combined = np.concatenate((x_test, x_test_pgd), axis=0)
    y_test_combined = np.concatenate((np.array([[1, 0]] * nb_test), np.array([[0, 1]] * nb_test)), axis=0)
    if model_type == 'svm' or model_type == 'randomforest':
        x_test_combined = x_test_combined.reshape(x_test_combined.shape[0], x_test_combined.shape[1] * x_test_combined.shape[2] * x_test_combined.shape[3])
        y_test_combined = np.argmax(y_test_combined, axis=1)


    elif model_type == 'logisticreg' or model_type == 'svm2' or model_type == 'rsf2':
        x_test_features = model.predict(x_test_combined, batch_size=64)
        x_test_combined = x_test_features.reshape((x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))
        y_test_combined = np.argmax(y_test_combined, axis=1)

    else:
        y_test_combined = np.argmax(y_test_combined, axis=1)

    acc2 = metrics.accuracy_score(y_test_combined, np.argmax((detector.predict(x_test_combined) > 0.5), axis=1))
    detector_acc_pgd += [acc2]
    print("ACC of detector on combined normal and pgd adversarial test set: %.2f%%" % (100 * acc2))

    # Evaluate acc of detector of combined normal and adversarial samples
    x_test_combined = np.concatenate((x_test, x_test_bim), axis=0)
    y_test_combined = np.concatenate((np.array([[1, 0]] * nb_test), np.array([[0, 1]] * nb_test)), axis=0)
    if model_type == 'svm' or model_type == 'randomforest':
        x_test_combined = x_test_combined.reshape(x_test_combined.shape[0], x_test_combined.shape[1] * x_test_combined.shape[2] * x_test_combined.shape[3])
        y_test_combined = np.argmax(y_test_combined, axis=1)


    elif model_type == 'logisticreg' or model_type == 'svm2' or model_type == 'rsf2':
        x_test_features = model.predict(x_test_combined, batch_size=64)
        x_test_combined = x_test_features.reshape((x_test_features.shape[0], x_test_features.shape[1] * x_test_features.shape[2] * x_test_features.shape[3]))
        y_test_combined = np.argmax(y_test_combined, axis=1)

    else:
        y_test_combined = np.argmax(y_test_combined, axis=1)

    acc3 = metrics.accuracy_score(y_test_combined, np.argmax((detector.predict(x_test_combined) > 0.5),axis=1))
    detector_acc_bim += [acc3]
    print("ACC of detector on combined normal and bim adversarial test set: %.2f%%" % (100 * acc3))
import pandas as pd
df = pd.DataFrame({"eps" : np.array(eps_range), "detect_fgsm" : np.array(detector_acc_fgsm), "detect_pgd": np.array(detector_acc_pgd), "detect_bim": np.array(detector_acc_bim),
                   "acc_fgsm": np.array(acc_fgsm), "acc_pgd": np.array(acc_pgd), "acc_bim": np.array(acc_bim)})
df.to_csv('/home/joelma/csv/' + dataset + '/' + model_type + '_detect_classify_acc_' + attack_type + str(epsilon) + '.csv', index=False)
