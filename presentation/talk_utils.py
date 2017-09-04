import numpy as np
import matplotlib.pylab as plt
import warnings
import numpy as np
from matplotlib import cm
# import pdb
from sklearn.metrics import confusion_matrix, accuracy_score, auc
import matplotlib.pyplot as plt
from talk_plottingutils import tile_raster_images, tile_raster_RGB


""""
-------------------------------------------------------------------------
toy data generation
------------------------
"""
def _create_circle_data(radius, std, n):
    r = np.random.normal(radius,std, size=n)
    angle = np.random.uniform(0,2*np.pi, size=n)
    x = np.cos(angle) * r
    y = np.sin(angle) * r
    return np.stack([x,y]).T

def create_nonlin_data(n):
    x1 = _create_circle_data(radius=0, std=1, n=n)
    x2 = _create_circle_data(radius=4, std=0.5, n=n)
    X = np.concatenate([x1,x2], axis=0)
    y = np.concatenate([np.zeros(len(x1)),
                        np.ones(len(x2))
                       ])
    return X , y 


def create_lin_data(n):
    x1 = np.random.multivariate_normal([-2, 0],0.5*np.eye(2), size=n)
    x2 = np.random.multivariate_normal([0, 2], 0.5*np.eye(2), size=n)
       
    X = np.concatenate([x1,x2], axis=0)
    y = np.concatenate([np.zeros(len(x1)),
                        np.ones(len(x2))
                       ])
    return X , y



""""
-------------------------------------------------------------------------
some ML functions
-------------------------------------------------------------------------
"""

def plot_confusion_matrix(y_true, y_hat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    cm = confusion_matrix(np.argmax(y_true,1), np.argmax(y_hat, 1))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_training(h_object):
    """
    h_object: what's returned by model.fit()
    """
    plt.plot(h_object.history['loss'])
    plt.plot(h_object.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')



