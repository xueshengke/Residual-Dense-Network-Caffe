# This file creates Residual Dense Network prototxt files: 'train_val' for training and 'deploy' for test
from __future__ import print_function
import sys
# caffe path
sys.path.append('/home/xueshengke/caffe-1.0/python')
from caffe import layers as L, params as P, to_proto
# from caffe.proto import caffe_pb2
import caffe

################################################################################
# change filename here
train_net_path = 'train_net.prototxt'
test_net_path = 'test_net.prototxt'
train_data_path = 'examples/RDN/train.txt'
test_data_path = 'examples/RDN/test.txt'

# parameters of the network
scale = 4
batch_size_train = 32
batch_size_test = 2
# number of RDBs
D = 6
# number of convolution layers per RDB
C = 6
# growth rate
G = 32
# number of feature maps outside RDBs
G_0 = 32
dropout = 0.0
################################################################################
# feature map size: output = (input - 1) * stride + kernel - 2 * pad
# x2: kernel=4, stride=2, pad=1;
# x3: kernel=5, stride=3, pad=1;
# x4: kernel=6, stride=4, pad=1;
def upscale(bottom, channel, scale, dropout):
    conv = L.Deconvolution(bottom, convolution_param=dict(
        num_output=channel, kernel_size=scale+2, stride=scale, pad=1, bias_term=False,
        weight_filler=dict(type='msra'), bias_filler=dict(type='constant')))
    relu = L.ReLU(conv, in_place=True)
    if dropout > 0:
        relu = L.Dropout(relu, dropout_ratio=dropout)
    return relu

def conv_relu(bottom, channel, kernel, stride, pad, dropout):
    conv = L.Convolution(bottom, num_output=channel, kernel_size=kernel, stride=stride, pad=pad,
                         bias_term=False, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'))
    relu = L.ReLU(conv, in_place=True)
    if dropout>0:
        relu = L.Dropout(relu, dropout_ratio=dropout)
    return relu

def add_layer(bottom, channel, dropout):
    conv = conv_relu(bottom, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)
    concate = L.Concat(bottom, conv, axis=1)
    return concate

def residual_dense_block(F_dn1, input_channel, depth, growth_rate, dropout):
    dense = F_dn1
    for i in range(depth):
        dense = add_layer(dense, channel=growth_rate, dropout=dropout)
    F_dLF = conv_relu(dense, channel=input_channel, kernel=1, stride=1, pad=0, dropout=dropout)
    F_d = L.Eltwise(F_dLF, F_dn1)
    return F_d

################################################################################
# define the network for training and validation
def train_RDN(train_data=train_data_path, test_data=test_data_path,
              batch_size_train=batch_size_train, batch_size_test=batch_size_test,
              block=D, depth=C, growth_rate=G, channel=G_0, dropout=dropout):
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': train_data, 'batch_size': batch_size_train}, include={'phase': caffe.TRAIN}, ntop=2)
    train_data_layer = str(net.to_proto())
    net.data, net.label = L.HDF5Data(hdf5_data_param={
        'source': test_data, 'batch_size': batch_size_test}, include={'phase': caffe.TEST}, ntop=2)

    net.F_n1 = conv_relu(net.data, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)
    net.F_0 = conv_relu(net.F_n1, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)

    net.block = net.F_0
    for i in range(block):
        net.block = residual_dense_block(net.block, input_channel=channel, depth=depth, growth_rate=growth_rate, dropout=dropout)
        if i == 0:
            last_concat = net.block
        else:
            net.contig_memory = L.Concat(last_concat, net.block, axis=1)
            last_concat = net.contig_memory
    net.global_fuse = conv_relu(net.contig_memory, channel=channel, kernel=1, stride=1, pad=0, dropout=dropout)
    net.F_GF = conv_relu(net.global_fuse, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)

    net.F_DF = L.Eltwise(net.F_GF, net.F_n1)
    net.deconv = upscale(net.F_DF, channel=channel, scale=scale, dropout=dropout)
    net.reconstruct = conv_relu(net.deconv, channel=3, kernel=3, stride=1, pad=1, dropout=dropout)
    net.loss = L.EuclideanLoss(net.reconstruct, net.label)

    return train_data_layer + str(net.to_proto())

################################################################################
# deploy the network for test; no data, label, loss layers
def test_RDN(block=D, depth=C, growth_rate=G, channel=G_0, dropout=dropout):
    net = caffe.NetSpec()

    net.data = L.Input(shape=dict(dim=[1,3,24,24]))
    net.F_n1 = conv_relu(bottom=net.data, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)
    net.F_0 = conv_relu(net.F_n1, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)

    net.block = net.F_0
    for i in range(block):
        net.block = residual_dense_block(net.block, input_channel=channel, depth=depth, growth_rate=growth_rate, dropout=dropout)
        if i == 0:
            last_concat = net.block
        else:
            net.contig_memory = L.Concat(last_concat, net.block, axis=1)
            last_concat = net.contig_memory
    net.global_fuse = conv_relu(net.contig_memory, channel=channel, kernel=1, stride=1, pad=0, dropout=dropout)
    net.F_GF = conv_relu(net.global_fuse, channel=channel, kernel=3, stride=1, pad=1, dropout=dropout)

    net.F_DF = L.Eltwise(net.F_GF, net.F_n1)
    net.deconv = upscale(net.F_DF, channel=channel, scale=scale, dropout=dropout)
    net.reconstruct = conv_relu(net.deconv, channel=3, kernel=3, stride=1, pad=1, dropout=dropout)
    # net.loss = L.EuclideanLoss(net.reconstruct, net.label)

    return net.to_proto()

################################################################################
if __name__ == '__main__':
    # write train_val network
    with open(train_net_path, 'w') as f:
        print(str(train_RDN()), file=f)
    print('create ' + train_net_path)

    # write test network
    with open(test_net_path, 'w') as f:
        f.write('name: "RDN_x' + str(scale) +'_block' + str(D) +'_depth' + str(C) +'_grow' + str(G) + '"\n')
        print(str(test_RDN()), file=f)
    print('create ' + test_net_path)
