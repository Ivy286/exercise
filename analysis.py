import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from Teemo.yuanpeng.yuanpeng_utils.read_excel import read_excel

from keras.layers import Input, Dense
from keras.models import Model


def build_data():
    '''

    :return: train_data, valid_data
    '''
    smiles_file_path = '/home/liyuanpeng/Desktop/EGFR/20/all.xlsx'
    column_name = ['ames_toxicity', 'cyp_3a4', 'cyp_2d6', 'cyp_2c9', 'bbb', 'p_glycoprotein',
                   'hia', 'renal_drug_to_drug', 'biodegradability', 'cell_permeability',
                   'tetrahymena_pyriformis', 'solubility']
    data = read_excel(smiles_file_path, column_name=column_name)

    data[9] = np.squeeze(np.array(data[9]))
    print (type(data))
    data[10] = np.squeeze(np.array(data[10], dtype=np.float32))
    train_data = [[x for x in vector[20:]] for vector in data]
    valid_data = [[x for x in vector[:20]] for vector in data]

    return train_data, valid_data

def nn_analysis(train_data, valid_data):

    inputs = Input(shape=(12,))
    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(64, activation='relu')(inputs)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    label_1 = [[float(1), float(0)]]*38
    label_0 = [[float(0), float(1)]]*16
    labels = np.array(label_1+label_0)
    train_data = np.array(train_data).T
    valid_data = np.array(valid_data).T
    # print (labels.shape)
    # print (np.array(train_data).shape)

    model.fit(train_data, labels, shuffle=True, epochs=1000, validation_split=0.18)
    all_predict = model.predict(valid_data)
    print ('all_predict:', np.argmax(all_predict, axis=-1))


def student_test(first_seq, second_seq):
    first_number_sample = first_seq.shape[0]
    second_number_sample = second_seq.shape[0]
    first_mean = np.mean(first_seq)
    first_std = np.std(first_seq)
    second_mean = np.mean(second_seq)
    second_std = np.std(second_seq)
    print ('first_seq_length:', len(first_seq))
    print ('second_seq_length:', len(second_seq))
    dev = 0.0
    dev2 = 0.0
    for value in second_seq:
        dev2 += np.square(value - second_mean)
    for value in first_seq:
        dev += np.square(value - first_mean)
    print ('first_mean:', first_mean)
    print ('second_mean:', second_mean)
    print ('first_std_sequare:', np.square(first_std))
    print ('second_std_sequare:', np.square(second_std))
    print ('dev:', dev)
    print ('dev2;', dev2)
    sw = np.sqrt((dev+dev2)/float(first_number_sample+second_number_sample-2))
    print ('sw', sw)
    t = (first_mean-second_mean)/(sw*(np.sqrt(1.0/float(first_number_sample)+1.0/float(second_number_sample))))
    return t

def statistic_analysis(train_data, valid_data):
    print (len(train_data))

    FDA = [[x for x in vector[:20]] for vector in train_data]

    terminated = [[x for x in vector[-11:]] for vector in train_data]


    # print (len(terminated[0]))
    # print (terminated[9])
    # print (FDA[9])
    bins = np.linspace(4.5, 7.0, 26)
    print (bins)
    plt.hist(FDA[9], color='g', bins=bins)
    plt.hist(terminated[9], color='r', alpha=0.6, bins=bins)
    plt.xlabel('cell permeability logPapp')
    plt.title('FDA compare terminated cell permeability')
    plt.show()

    print ('terminated cell permeability: {0}, FDA cell permeability: {1}'.format(np.mean(terminated[9]),
                                                                                  np.mean(FDA[9])))
    print ('terminated tetrahymena_pyriformis: {0}, FDA tetrahymena_pyriformis: {1}'.format(np.mean(terminated[10]),
                                                                                            np.mean(FDA[10])))
    print ('terminated solubility: {0}, FDA solubility: {1}'.format(np.mean(terminated[11]),
                                                                    np.mean(FDA[11])))
    print ('cell permeability', np.mean(valid_data[9]))
    t_test_solubility = student_test(np.array(FDA[11]), np.array(terminated[11]))
    t_test_cell_permeability = student_test(np.array(FDA[9]), np.array(terminated[9]))
    print ('t_test_cell_permeability:', t_test_cell_permeability)
    print ('t_test_solubility:', t_test_solubility)

def pca_analysis(train_data, valid_data):

    # FDA = np.array([[x for x in vector[:20]] for vector in train_data])
    #
    # terminated = np.array([[x for x in vector[-11:]] for vector in train_data])
    #
    # print ('FDA:', FDA.shape)
    # print ('terminated:', terminated.shape)
    #
    # fit_sklearn_input_FDA = FDA.T
    # fit_sklearn_input_terminated = terminated.T

    all_data = np.array([[x for x in vector] for vector in train_data])

    fit_sklearn_input_data = all_data.T

    pca = PCA(n_components=2)
    pca.fit(fit_sklearn_input_data)

    print(pca.explained_variance_ratio_)

    y = pca.transform(fit_sklearn_input_data)
    print ('y:', y)

    FDA_pca_data = y[:20].T
    print (FDA_pca_data.shape)
    terminated = y[-11:].T
    print (terminated.shape)

    plt.scatter(FDA_pca_data[0], FDA_pca_data[1], color='r', marker='*')

    plt.scatter(terminated[0], terminated[1], color='g', marker='o')
    plt.show()

    # all_list = read_excel('/home/liyuanpeng/Desktop/fda_terminated.xlsx', column_name=['col 1', 'col 2'])

    # print (all_list)
#
#
# def plot_decision_function(X, classifier, sample_weight, axis, title):
#     xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
#
#     Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
#     axis.scatter(X[:, 0], X[:, 1], c=yy, s=100 * sample_weight, alpha=0.9)
#     axis.axis('off')
#     axis.set_title(title)


def svm_analysis(train_data, valid_data):

    all_data = np.array([[x for x in vector] for vector in train_data])

    solubility = all_data[11]
    # plt.hist(solubility, bins=20)
    # plt.show()

    fit_sklearn_input_data = all_data.T

    pca = PCA(n_components=2)
    pca.fit(fit_sklearn_input_data)
    x = pca.transform(fit_sklearn_input_data)
    fit_svm_x = list(x[:20]) + list(x[-11:])

    y = [1]*20 + [0]*11

    print ('y:', y)
    print ('fit_sklearn_input_data:', fit_svm_x)
    clf = svm.LinearSVC()

    print ('fit_svm_x:', len(fit_svm_x))
    clf.fit(fit_svm_x, y)
    fit_svm_x = np.array(fit_svm_x)

    lw = clf.coef_[0]
    la = -lw[0]/lw[1]
    xx = np.linspace(-5, 5)
    ly = la*xx - clf.intercept_[0]/lw[1]


    print ('valid_data:', np.array(valid_data).shape)
    print ('fit_sklearn_input_data', fit_sklearn_input_data.shape)
    fit_sklearn_input_validdata = np.array(valid_data).T
    print ('valid_data:', fit_sklearn_input_validdata.shape)
    pca_result = pca.transform(fit_sklearn_input_validdata)
    print (pca_result.shape)


    all_predict = clf.predict(fit_svm_x)
    print (all_predict)

    linear_confusion_matrix = confusion_matrix(y, all_predict)
    print ('linear_confusion matrix:', linear_confusion_matrix)




    plt.plot(xx, ly, 'k-')
    plt.scatter(fit_svm_x[:20, 0], fit_svm_x[:20, 1], color='g', label='On-going')
    plt.scatter(fit_svm_x[-11:, 0], fit_svm_x[-11:, 1], color='r', label='Terminated')
    # plt.scatter(pca_result[:, 0], pca_result[:, 1], color='g', label='20 smiles')
    # plt.scatter(pca_result[7, 0], pca_result[7, 1], marker='*', label='azd9291')
    #plt.scatter(pca_result[])
    #plt.title('admet pca add svm classification')
    plt.legend(loc='upper right', shadow=True,  markerscale=2, scatterpoints=1)
    plt.show()

    # plt.legend.get_frame().set_facecolor('#00FFCC')

    # weights = np.ones(len(fit_svm_x))
    #
    # fig, axis = plt.subplots(1, 1)
    #
    # plt.plot(xx, ly, 'k-')
    #
    # fit_svm_x = np.array(fit_svm_x)
    #
    # plot_decision_function(fit_svm_x, clf, weights, axis, 'nonlinear')
    # plt.show()


def nonlinear_svm_analysis(train_data, valid_data):
    all_data = np.array([[x for x in vector] for vector in train_data])

    # solubility = all_data[11]
    # plt.hist(solubility, bins=20)
    # plt.show()

    fit_sklearn_input_data = all_data.T
    pca = PCA(n_components=2)

    pca.fit(fit_sklearn_input_data)
    x = pca.transform(fit_sklearn_input_data)
    fit_svm_x = list(x[:10]) + list(x[-11:])


    y = [1] * 10 + [0] * 11

    # print ('y:', y)
    # print ('fit_sklearn_input_data:', fit_svm_x)

    clf = svm.SVC()

   # print ('fit_svm_x:', len(fit_svm_x))
    clf.fit(fit_svm_x, y)

    all_predict = clf.predict(fit_svm_x)
    #print (all_predict)

    nonlinear_confusion_matrix = confusion_matrix(y, all_predict)
    #print ('nonlinear_confusion matrix:', nonlinear_confusion_matrix)

    fit_sklearn_input_validdata = np.array(valid_data).T
    print ('azd9291:', fit_sklearn_input_validdata[7])
    pca_result = pca.transform(fit_sklearn_input_validdata)
    print (pca_result[7])

    azd_9291 = pca_result[7]

    weights = np.ones(len(fit_svm_x))

    fig, axis = plt.subplots(1, 1)

    fit_svm_x = np.array(fit_svm_x)

    plot_decision_function(fit_svm_x, clf, weights, axis, 'nonlinear', azd_9291=azd_9291)
    plt.legend(loc='upper right', scatterpoints=1, fontsize='small')

    plt.show()

def plot_decision_function(X, classifier, sample_weight, axis, title, azd_9291=None):

    xx, yy = np.meshgrid(np.linspace(-5, 5, 501), np.linspace(-2, 3, 501))

    print ('xx:', xx)
    print ('yy:', yy)
    print ('exercise:', len(np.c_[xx.ravel(), yy.ravel()]))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

    print ('Z:', Z)

    Z = Z.reshape(xx.shape)
    plt.plot()
    axis.contourf(xx, yy, Z, 10, alpha=0.6)
    axis.scatter(X[:10, 0], X[:10, 1], color='g', s=100*sample_weight, label='On-going')
    axis.scatter(X[-11:, 0], X[-11:, 1], color='r', s=100*sample_weight, label='Terminated')

    if azd_9291 != None:
        axis.scatter(azd_9291[0], azd_9291[1], color='black', s=100 * sample_weight, marker='*')
    axis.axis('off')
    axis.set_title(title)
    # plt.plot(axis)

    # print ('num sv', len(clf.support_))
    # print ('num sv', clf.support_)
    #
    # negative_support_vector = clf.support_[:11]
    # positive_support_vector = clf.support_[11:]

    # print (fit_svm_x[negative_support_vector, 0])




    # plt.scatter(fit_svm_x[:20, 0], fit_svm_x[:20, 1], color='y')
    # # plt.scatter(fit_svm_x[-11:, 0], fit_svm_x[-11:, 1])
    #
    # plt.scatter(fit_svm_x[negative_support_vector, 0], fit_svm_x[negative_support_vector, 1], color='b')
    # plt.scatter(fit_svm_x[positive_support_vector, 0], fit_svm_x[positive_support_vector, 1], color='r')
    # #plt.scatter()
    # plt.show()


    # weight = np.ones(len(fit_svm_x))
    # fig, axis = plt.subplots(1, 1)
    # plot_decision_function(x, clf, weight, axis, 'exercise')



if __name__ == '__main__':

    train_data, valid_data = build_data()
    # train_data = np.array(train_data).T
    # print (train_data.shape)
    # nn_analysis(train_data, valid_data)

    #statistic_analysis(train_data, valid_data)

    #svm_analysis(train_data, valid_data)

    nonlinear_svm_analysis(train_data, valid_data)






    #pca_analysis(train_data, valid_data)


    # x = np.array([76.43, 76.21, 73.58, 69.69, 65.29, 70.83, 82.75, 72.34])
    # y = np.array([73.66, 64.27, 69.34, 71.37, 69.77, 68.12, 67.27, 68.07, 62.61])
    # t = student_test(x, y)
    # print (t)



