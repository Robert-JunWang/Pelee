from __future__ import print_function

import os, math

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def _conv_block(net, bottom, name, num_output, use_relu=True, kernel_size=3, stride=1, pad=1, bn_prefix='', bn_postfix='/bn', 
    scale_prefix='', scale_postfix='/scale'):

    conv = L.Convolution(bottom, kernel_size=kernel_size, stride=stride, 
                    num_output=num_output,  pad=pad, bias_term=False, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    net[name] = conv

    bn_name = '{}{}{}'.format(bn_prefix, name, bn_postfix)
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': 0.001,
        'moving_average_fraction': 0.999,
        }
    batch_norm = L.BatchNorm(conv, in_place=True, **bn_kwargs)
    net[bn_name] = batch_norm

    scale = L.Scale(batch_norm, bias_term=True, in_place=True, filler=dict(value=1), bias_filler=dict(value=0))
    sb_name = '{}{}{}'.format(scale_prefix, name, scale_postfix)
    net[sb_name] = scale

    if use_relu:
        out_layer = L.ReLU(scale, in_place=True)
        relu_name = '{}/relu'.format(name)
        net[relu_name] = out_layer
    else:
        out_layer = scale

    return out_layer

def _dense_block(net, from_layer, num_layers, growth_rate, name,bottleneck_width=4):

  x = from_layer
  growth_rate = int(growth_rate/2)

  for i in range(num_layers):
    base_name = '{}_{}'.format(name,i+1)
    inter_channel = int(growth_rate * bottleneck_width / 4) * 4

    cb1 = _conv_block(net, x, '{}/branch1a'.format(base_name), kernel_size=1, stride=1, 
                               num_output=inter_channel, pad=0)
    cb1 = _conv_block(net, cb1, '{}/branch1b'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)

    cb2 = _conv_block(net, x, '{}/branch2a'.format(base_name), kernel_size=1, stride=1, 
                               num_output=inter_channel, pad=0)
    cb2 = _conv_block(net, cb2, '{}/branch2b'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)
    cb2 = _conv_block(net, cb2, '{}/branch2c'.format(base_name), kernel_size=3, stride=1, 
                               num_output=growth_rate, pad=1)

    x = L.Concat(x, cb1, cb2, axis=1)
    concate_name = '{}/concat'.format(base_name)
    net[concate_name] = x

  return x



def _transition_block(net, from_layer, num_filter, name, with_pooling=True):

  conv = _conv_block(net, from_layer, name, kernel_size=1, stride=1, num_output=num_filter, pad=0)

  if with_pooling:
    pool_name = '{}/pool'.format(name)
    pooling = L.Pooling(conv, pool=P.Pooling.AVE, kernel_size=2, stride=2)
    # pooling = L.Pooling(conv, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net[pool_name] = pooling
    from_layer = pooling
  else:
    from_layer = conv


  return from_layer



def _stem_block(net, from_layer, num_init_features):

  stem1 = _conv_block(net, net[from_layer], 'stem1', kernel_size=3, stride=2,
                           num_output=num_init_features, pad=1)
  stem2 = _conv_block(net, stem1, 'stem2a', kernel_size=1, stride=1,
                           num_output=int(num_init_features/2), pad=0)
  stem2 = _conv_block(net, stem2, 'stem2b', kernel_size=3, stride=2,
                           num_output=num_init_features, pad=1)
  stem1 = L.Pooling(stem1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
  net['stem/pool'] = stem1

  concate = L.Concat(stem1, stem2, axis=1)
  concate_name = 'stem/concat'
  net[concate_name] = concate

  stem3 = _conv_block(net, concate, 'stem3', kernel_size=1, stride=1, num_output=num_init_features, pad=0)

  return stem3

def PeleeNetBody(net, from_layer='data', growth_rate=32, block_config = [3,4,8,6], bottleneck_width=[1,2,4,4], num_init_features=32, init_kernel_size=3, use_stem_block=True):

    assert from_layer in net.keys()

    # Initial convolution
    if use_stem_block:
      from_layer = _stem_block(net, from_layer, num_init_features)

    else:
      padding_size = init_kernel_size / 2
      out_layer = _conv_block(net, net[from_layer], 'conv1', kernel_size=init_kernel_size, stride=2,
                               num_output=num_init_features, pad=padding_size)
      net.pool1 = L.Pooling(out_layer, pool=P.Pooling.MAX, kernel_size=2, pad=0,stride=2)
      from_layer = net.pool1

    total_filter = num_init_features
    if type(bottleneck_width) is list:
        bottleneck_widths = bottleneck_width
    else:
        bottleneck_widths = [bottleneck_width] * 4

    for idx, num_layers in enumerate(block_config):
      from_layer = _dense_block(net, from_layer, num_layers, growth_rate, name='stage{}'.format(idx+1), bottleneck_width=bottleneck_widths[idx])
      total_filter += growth_rate * num_layers
      
      if idx == len(block_config) - 1:
        with_pooling=False
      else:
        with_pooling=True

      from_layer = _transition_block(net, from_layer, total_filter,name='stage{}_tb'.format(idx+1), with_pooling=with_pooling)



    return net


def add_classify_header(net, classes=120):
  bottom = net.keys()[-1]

  net.global_pool = L.Pooling(net[bottom], pool=P.Pooling.AVE, global_pooling=True) 

  net.classifier = L.InnerProduct(net.global_pool, num_output=classes, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))

  net.prob = L.Softmax(net.classifier)
  return net



if __name__ == '__main__':
  net = caffe.NetSpec()
  net.data = L.Input(shape=[dict(dim=[1, 3, 224, 224])])

  PeleeNetBody(net, from_layer='data')
  add_classify_header(net,classes=1000)
  print(net.to_proto())


