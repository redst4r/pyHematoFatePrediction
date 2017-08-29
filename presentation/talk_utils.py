import numpy as np
import matplotlib.pylab as plt
import warnings
import numpy as np
from matplotlib import cm
# import pdb

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
from sklearn.metrics import confusion_matrix, accuracy_score, auc
import matplotlib.pyplot as plt
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



def tile_raster_images(X, img_dim, tile_shape, tile_spacing=(1, 1),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=True,
                       cmap=cm.gray,
                       img_shape=None,
                       figsize=None):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type img_dim: tuple; (height dim, width dim)
    :param img_dim: which dimensions contain the img spatial coordinates

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    if img_shape:
        warnings.warn('img_shape is decpreacted. use img_dim')
    else:
        assert img_dim
        assert max(
            img_dim) < X.ndim, "X has only %d dimensions, you specified spatial dimensions %d,%d.  Maybe some relicate from img_shape.img_dim" % (
        X.ndim, img_dim[0], img_dim[1])

        """
        the entire code below assumes that channels are in the first dimension, then space, i.e. (c,x,y)
        lets just make it that way
        """
        channel_dim = [_ for _ in range(3) if _ not in img_dim]
        assert len(channel_dim) == 1
        channel_dim = channel_dim[0]
        X = X.transpose([channel_dim, img_dim[0], img_dim[1]])

        # now space is in dim 1,2
        img_shape = X.shape[1:]

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                                              dtype='uint8' if output_pixel_vals else out_array.dtype
                                              ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing,
                                                        scale_rows_to_unit_interval, output_pixel_vals)

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

        # make the entire ouput between 0,1.  the FLAG scale_rows_to_unit_interval does this for each sample independently!!
        X = _scale_to_unit_interval(X)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = _scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] \
                        = this_img * (255 if output_pixel_vals else 1)
        if figsize:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        plt.imshow(out_array, interpolation='nearest', cmap=cmap)

        # return out_array


def tile_raster_RGB(X, tile_shape, tile_spacing, scale_rows_to_unit_interval=False):
    assert X.shape[3] == 3, 'works only for three channels'
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    img_shape = X.shape[1:3]

    H, W = img_shape
    Hs, Ws = tile_spacing

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]
    out_array = np.zeros(out_shape + [3], dtype=X.dtype)

    X = _scale_to_unit_interval(X)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = _scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col])
                else:
                    this_img = X[tile_row * tile_shape[1] + tile_col]
                # add the slice to the corresponding position in the
                # output array
                out_array[tile_row * (H + Hs): tile_row * (H + Hs) + H, tile_col * (W + Ws): tile_col * (W + Ws) + W, :] \
                    = this_img
    plt.figure()
    plt.imshow(out_array, interpolation='nearest')


def _scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= np.nanmin(ndar)
    ndar *= 1.0 / (np.nanmax(ndar) + eps)
    return ndar