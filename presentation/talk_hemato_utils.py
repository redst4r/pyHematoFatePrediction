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


def load_data(pickle_file, N, randomize=True):
    # pickle_file = '../images_round3_test_annotated.pickle'
    res = load_pickle(pickle_file)
    X = []
    y = []
    movement = []
    cellIDs = []
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
    # shuffle the datapoints anyway, such that timepoints of the same cell wont be consecutive
    ix_shuffle = np.random.permutation(X.shape[0])
    X = X[ix_shuffle]
    y = y[ix_shuffle]
    cellIDs = np.stack(cellIDs)
    
    return X, y, movement, cellIDs
