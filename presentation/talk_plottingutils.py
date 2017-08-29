import numpy as np
import bokeh.plotting as bp
import holoviews as hv
import matplotlib.pyplot as plt
import cv2

""" tools for plotting images in the ipython notebook """


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


def plot_image_bokeh(img, points=None):
    # analog of matplotlib.pyplot.imshow()
    # optinally plots a cloud of points ontop of the image as a scatter
    p = bp.figure(x_range=(0, img.shape[1]), y_range=(0, img.shape[0]))
    cmap ='Greys256'

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
