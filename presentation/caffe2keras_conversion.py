import h5py
import numpy as np

def load_weights_caffe2keras(caffe_h5_weights, CNN, bn_trainable=True, other_param_trainable=True):
    # caffe_h5_weights = '../outfile.hdf5'
    # load the weights from the caffe model hdf5
    f = h5py.File(caffe_h5_weights, "r")

    def rot180(W):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = np.rot90(W[i, j], 2)
        return W


    def conv_caffe2keras(W):
        # return rot180(W).transpose([2,3,1,0])  # claimed to be needed due to caffe-correlation vs keras-conv, but aparantyly keras handles it correcled alreay
        return W.transpose([2,3,1,0])

    conv1W = conv_caffe2keras(f['/data/conv1/0'].value)
    conv1b = f['/data/conv1/1'].value

    # for batchnorm docs:
    # hese statistics are kept in the layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
    conv1_BN_mean = f['/data/conv1_BN/0'].value
    conv1_BN_var = f['/data/conv1_BN/1'].value


    conv2W = conv_caffe2keras(f['/data/conv2/0'].value)
    conv2b = f['/data/conv2/1'].value

    conv2_BN_mean = f['/data/conv2_BN/0'].value
    conv2_BN_var = f['/data/conv2_BN/1'].value


    conv3W = conv_caffe2keras(f['/data/conv3/0'].value)
    conv3b = f['/data/conv3/1'].value

    conv3_BN_mean = f['/data/conv3_BN/0'].value
    conv3_BN_var = f['/data/conv3_BN/1'].value

    fc6W = f['/data/fc6/0'].value.transpose()
    fc6b = f['/data/fc6/1'].value
    fc6_BN_mean = f['/data/fc6_BN/0'].value
    fc6_BN_var = f['/data/fc6_BN/1'].value

    fc7W = f['/data/fc7/0'].value.transpose() # [:-1] # skip the last row, which is how the speed enters the network
    fc7b = f['/data/fc7/1'].value
    fc7_BN_mean = f['/data/fc7_BN/0'].value
    fc7_BN_var = f['/data/fc7_BN/1'].value

    fc8W = f['/data/fc8/0'].value.transpose()
    fc8b = f['/data/fc8/1'].value

    f.close()

    # put those weights on the layers and freeze them
    CNN.get_layer('conv1').set_weights([conv1W, conv1b])
    CNN.get_layer('conv1').trainable = other_param_trainable

    CNN.get_layer('conv2').set_weights([conv2W, conv2b])
    CNN.get_layer('conv2').trainable = other_param_trainable

    CNN.get_layer('conv3').set_weights([conv3W, conv3b])
    CNN.get_layer('conv3').trainable = other_param_trainable

    CNN.get_layer('fc6').set_weights([fc6W, fc6b])
    CNN.get_layer('fc6').trainable = other_param_trainable

    CNN.get_layer('fc7').set_weights([fc7W, fc7b])
    CNN.get_layer('fc7').trainable = other_param_trainable

    CNN.get_layer('fc8').set_weights([fc8W, fc8b])
    CNN.get_layer('fc8').trainable = other_param_trainable

    # the batchnorm layers are handled differently in keras/caffe
    def BN_caffe2keras(caffe_mean, caffe_var):
        nb_kernels = caffe_mean.shape[0]
        return  [#np.ones(nb_kernels), 
                 #np.zeros(nb_kernels), 
                 caffe_mean.astype(dtype=np.float32), 
                 caffe_var.astype(dtype=np.float32)]

    CNN.get_layer('conv1_BN').set_weights(BN_caffe2keras(conv1_BN_mean, conv1_BN_var))
    CNN.get_layer('conv1_BN').trainable = bn_trainable

    CNN.get_layer('conv2_BN').set_weights(BN_caffe2keras(conv2_BN_mean, conv2_BN_var))
    CNN.get_layer('conv2_BN').trainable = bn_trainable

    CNN.get_layer('conv3_BN').set_weights(BN_caffe2keras(conv3_BN_mean, conv3_BN_var))
    CNN.get_layer('conv3_BN').trainable = bn_trainable

    CNN.get_layer('fc6_BN').set_weights(BN_caffe2keras(fc6_BN_mean, fc6_BN_var))
    CNN.get_layer('fc6_BN').trainable = bn_trainable

    CNN.get_layer('fc7_BN').set_weights(BN_caffe2keras(fc7_BN_mean, fc7_BN_var))
    CNN.get_layer('fc7_BN').trainable = bn_trainable


    CNN.compile(optimizer='adam', loss='categorical_crossentropy')

    return CNN

if __name__ == '__main__':

    CNN = load_weights_caffe2keras('/mnt/outfile.hdf5', CNN)

    #from talk_hemato_utils import load_pickle
    from talk_hemato_utils import load_data
    X,y  = load_data('/mnt/images_round3_test_annotated.pickle', N=10, randomize=True)

    # look at the acivations after the first conv in keras and caffe
    X0 = X[:1]

    from keras.models import Model
    c1 = Model(CNN.input, CNN.get_layer('conv1_act').output)
    A1 = c1.predict(X0)


    # CAFFE
    net.blobs['data0'].data[...] = X0.transpose([0,3,1,2])
    net.forward()
    A2 = net.blobs['conv1'].data.transpose([0,2,3,1]) 





    # lets check after first batchnorm

    bn1 = Model(CNN.input, CNN.get_layer('conv1_BN').output)
    bn1_layer = bn1.get_layer('conv1_BN')
    X0 = X[:1]
    B1 = bn1.predict(X0)
    the_mean, the_var = bn1_layer.get_weights()

    # looks like keras is actually normalizing by the sqrt(var)
    # if i skip the sqrt, B1 and B1hat dont match even closely
    B1hat = A1/(np.sqrt(the_var)+1e-3) - the_mean/(np.sqrt(the_var)+1e-3)


    net.blobs['data0'].data[...] = X0.transpose([0,3,1,2])
    net.forward()

    B2 = net.blobs['conv1_BN'].data.transpose([0,2,3,1]) 


    # load a minibatch
    net.blobs['data0'].data = net.blobs['data0'].data.reshape(2,1,27,27)
    net.blobs['data0'].data[...] = X[:2].transpose([0,3,1,2])