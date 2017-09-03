import pickle
import numpy as np
import keras
import sys

def load_pickle(pickle_file):
    fpickle = open(pickle_file, 'rb')

    if sys.version_info >= (3,0):
        input_image = pickle.load(fpickle, encoding='latin1')
        lab = pickle.load(fpickle, encoding='latin1')
        mov = pickle.load(fpickle, encoding='latin1')
        cellID = pickle.load(fpickle, encoding='latin1')
    else:
        input_image = pickle.load(fpickle)
        lab = pickle.load(fpickle)
        mov = pickle.load(fpickle)
        cellID = pickle.load(fpickle)


    fpickle.close()

    res = {}
    res['mov'] = mov
    res['im'] = input_image
    res['label'] = lab
    res['cellIDs'] = 'Cell'+cellID[0][0].split('_')[5][4:]
    return res


def load_data(pickle_file, N=None, randomize=True):
    # pickle_file = '../images_round3_test_annotated.pickle'
    res = load_pickle(pickle_file)
    X = []
    y = []
    movement = []
    cellIDs = []

    if not N:
        N = len(res['im'])
    
    cell_indices = np.random.permutation(len(res['im']))[:N] if randomize else range(min(N, len(res['im'])))

    for i in cell_indices:
        img_array = res['im'][i]  # an array of images of that cell over time, a single element is 27x27x1
        the_label = res['label'][i]  # a single label of that cell
        mov = res['mov'][i] 
        label_array = [the_label for _ in range(len(img_array))]         # extend the label for the entire cell
        cellid_array = [i for _ in range(len(img_array))]         # extend the cellid for the entire cell

        X.extend(img_array)
        y.extend(label_array)
        movement.extend(mov)
        cellIDs.extend(cellid_array)

    X = np.stack(X) / 256  # (sample,27,27,1
    y = np.stack(y)  # (sample, 
    y = keras.utils.to_categorical(y, num_classes=2)
    movement = np.concatenate(movement)
    cellIDs = np.stack(cellIDs)

    # shuffle the datapoints anyway, such that timepoints of the same cell wont be consecutive
    ix_shuffle = np.random.permutation(X.shape[0])
    X = X[ix_shuffle]
    y = y[ix_shuffle]
    movement = movement[ix_shuffle]
    cellIDs = cellIDs[ix_shuffle]
    return X, y, movement, cellIDs


def create_hemato_cnn():
    from keras.models import Model
    from keras.layers import Input, Dense, Conv2D, Dropout, MaxPool2D, BatchNormalization, Activation, Flatten, Concatenate

    padding = 'valid'
    mp_padding = 'same' # that how its implemented in the original caffe model
    bn_scaling = False  # caffemodel didnt have any scaling (ok, since rectiviers) 
    bn_offset = False  # caffemodel didnt have any offset (ok, since rectiviers)

    # Functional API

    # Input layers for the image and cell-speed
    img_input = Input((27,27,1), name='img_input') # names as in  caffe
    mov_input = Input((1,), name='mov_input')

    # conv1 
    c1 = Conv2D(filters=20, kernel_size=(5,5), activation='linear', padding=padding, name='conv1')(img_input)
    c1A = Activation('relu', name='conv1_act')(c1)
    c1BN = BatchNormalization(scale=bn_scaling, center=bn_offset, name='conv1_BN')(c1A)
    c1MP = MaxPool2D(pool_size=(2,2), name='pool1', padding=mp_padding)(c1BN)

    # conv2 
    c2 = Conv2D(filters=60, kernel_size=(4,4), activation='linear', padding=padding, name='conv2')(c1MP)
    c2A = Activation('relu', name='conv2_act')(c2)
    c2BN =BatchNormalization(scale=bn_scaling, center=bn_offset, name='conv2_BN')(c2A)
    c2MP = MaxPool2D(pool_size=(2,2), name='pool2', padding=mp_padding)(c2BN)

    # conv3
    c3 = Conv2D(filters=100, kernel_size=(3,3), activation='linear', padding=padding, name='conv3')(c2MP)
    c3A = Activation('relu', name='conv3_act')(c3)
    c3BN = BatchNormalization(scale=bn_scaling,center=bn_offset, name='conv3_BN')(c3A)
    c3MP = MaxPool2D(pool_size=(2,2), name='pool5', padding=mp_padding)(c3BN)

    # fc6
    flat_layer = Flatten()(c3MP)
    fc6 = Dense(units=500, activation='linear', name='fc6')(flat_layer)
    fc6A = Activation('relu', name='fc6_act')(fc6)

    # concat with mov
    mov1 = Dense(units=1, activation='relu', name='ipm0')(mov_input)
    concat = Concatenate(axis=-1)([fc6A, mov1])

    fc6BN = BatchNormalization(scale=bn_scaling,center=bn_offset, name='fc6_BN')(concat)
    fc6D = Dropout(rate=0.5, name='drop6')(fc6BN)

    # fc7
    fc7= Dense(units=50, activation='linear', name='fc7')(fc6D)
    fc7A=Activation('relu', name='fc7_act')(fc7)
    fc7BN = BatchNormalization(scale=bn_scaling, center=bn_offset, name='fc7_BN')(fc7A)
    fc7D = Dropout(rate=0.5, name='drop7')(fc7BN)

    # fc8 / softmax
    fc8 = Dense(units=2, activation='linear', name='fc8')(fc7D)
    fc8A = Activation('softmax', name='fc8_act')(fc8)

    CNN = Model(inputs=[img_input, mov_input], outputs=fc8A)
    return CNN


def get_auc(scores, truelabel, do_plot=False):
    from sklearn.metrics import roc_curve, auc, accuracy_score
    fpr, tpr, thresholds = roc_curve(truelabel, scores, drop_intermediate=False)
    the_auc = auc(fpr, tpr)
    if do_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(fpr, tpr, c=thresholds, cmap=plt.cm.bwr)
        plt.title("AUC %.02f" % the_auc)
        plt.plot(fpr,tpr,'k')
        plt.xlim([0,1])
        plt.ylim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.colorbar()
    return the_auc


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]
