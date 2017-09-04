import numpy as np
import bokeh.plotting as bp
import holoviews as hv
import matplotlib.pyplot as plt
import cv2

""" tools for plotting images in the ipython notebook """
def normalize01(x):
    'scale x into [0,1]'
    return (x- x.min())/ (x- x.min()).max()



def hv_plot_stack(img_list, timepoints=None):
    if not timepoints:
        timepoints = range(len(img_list))

    dictionary = {int(t): hv.Image(arr, bounds=None, kdims=['x', 'y'])
                  for t, arr in zip(timepoints, img_list)}

    return hv.HoloMap(dictionary, kdims=['Time'])


"""
-----------------------------------------------------------------------------------------
BOKEH stuff
-----------------------------------------------------------------------------------------
"""

def plot_image_bokeh(img, points=None, cmap='Greys256'):
    # analog of matplotlib.pyplot.imshow()
    # optinally plots a cloud of points ontop of the image as a scatter

    # to comply with thet usual imshow (origin in the upper left corner), flip the image along the row axis
    img = img[::-1,:]
    if points is not None:
        n_rows = img.shape[0]  # the y-axis
        points[:,1] = n_rows - points[:,1]  # also invert the scatter


    p = bp.figure(x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))

    p.image(image=[img], x=0, y=0, dw=img.shape[1], dh=img.shape[0], palette=cmap)

    if points is not None:
        p.scatter(points[:,0], points[:,1])

    bp.show(p)

    return p


def plot_image_rgb_bokeh(I):
    "display a RGB-array in bokeh. some ridiculous overhead to get it into displayable format..."

    the_image = I.astype('uint8')  # this supid bokeh thing where you have to compress color channels into 32bit encoding...

    if I.ndim == 3:
        # add alpha channel
        the_image = np.dstack([the_image, np.ones(the_image.shape[:2], np.uint8) * 255])

    the_image = np.squeeze(the_image.view(np.uint32))

    # to comply with imshow, flip the y-axis
    the_image = the_image[::-1]

    p = bp.figure(x_range=(0, the_image.shape[1]), y_range=(0, the_image.shape[0]))
    p.image_rgba([the_image], x=0, y=0, dw=the_image.shape[1], dh=the_image.shape[0])
    bp.show(p)

    return p


def plot_segmentation_bokeh(I, I_mask):
    
    merged = merge_image_and_seg(I, I_mask)
    plot_image_rgb_bokeh(merged)


"""
-----------------------------------------------------------------------------------------
matplotlib stuff
-----------------------------------------------------------------------------------------
"""


def plot_image_mpl(img):
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.colorbar()


def plot_segmentation(I, I_mask):
    plt.imshow(I, cmap=plt.cm.Greys_r)
    plt.imshow(I_mask, cmap=plt.cm.Reds, alpha=0.5)


from mpl_toolkits.mplot3d import Axes3D

def plot_3d(x, y):
    "x: the data.  c: optinal coloring of datapoints"
    # WARNINI currently only for two classes
    fig = plt.figure()
    #     fig.clf()
    ax = Axes3D(fig)
    for i,c in zip([0,1], ['red', 'blue']):
        ix = y==i
        ax.plot(x[ix, 0], 
                x[ix, 1], 
                x[ix, 2], 'o', c=c, alpha=0.5)# label=my_labels[index]


plt.show()
"""
-----------------------------------------------------------------------------------------
gereral stuff
-----------------------------------------------------------------------------------------
"""

def merge_image_and_seg(I, I_mask):
    "overlay the image and a segmentation mask. returns a RGB image array"

    alpha = 0.5
    # first convert them into RGB images
    Ic = np.stack([I, I, I], axis=-1)
    seg_mask_c = np.zeros(I_mask.shape + (3,))
    # important to make 0,1 -> 0,255 otherwise a 1pixel value wont be visible
    seg_mask_c[:,:,0] = I_mask * 255
    output = cv2.addWeighted(seg_mask_c.astype('uint8'), alpha, Ic, 1-alpha, 0)
    return output


def patch_box(x,y ,X_WINDOWSIZE_HALF, Y_WINDOWSIZE_HALF, imgShape ):

    if x-X_WINDOWSIZE_HALF < 1:
        xpad_left = 1+np.abs(x-X_WINDOWSIZE_HALF)
        xstart = 1
    else:
        xstart = x-X_WINDOWSIZE_HALF
        xpad_left = 0

    if x+X_WINDOWSIZE_HALF > imgShape[1]:
        xpad_right = x+X_WINDOWSIZE_HALF - imgShape[1]
        xend = imgShape[1]
    else:
        xpad_right = 0
        xend = x+X_WINDOWSIZE_HALF

    ## y

    if y-Y_WINDOWSIZE_HALF < 1:
        ypad_top = 1+np.abs(y-Y_WINDOWSIZE_HALF)
        ystart = 1
    else:
        ystart = y-Y_WINDOWSIZE_HALF
        ypad_top = 0

    if y+Y_WINDOWSIZE_HALF > imgShape[0]:
        ypad_bottom = y+Y_WINDOWSIZE_HALF - imgShape[0]
        yend = imgShape[0]
    else:
        ypad_bottom = 0
        yend = y+Y_WINDOWSIZE_HALF

    assert all([xstart>0, xend>0, ystart>0,yend>0 ])
    assert all([xpad_left>=0, xpad_right>=0, ypad_top>=0,ypad_bottom>=0 ])

    return (xstart, xend),(ystart, yend),  (xpad_left, xpad_right), (ypad_top, ypad_bottom)



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


def tile_raster_RGB(X, tile_shape, tile_spacing, scale_rows_to_unit_interval=False, figsize=None):
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
    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.imshow(out_array, interpolation='nearest')


def _scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= np.nanmin(ndar)
    ndar *= 1.0 / (np.nanmax(ndar) + eps)
    return ndar