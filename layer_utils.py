import os, math

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def res_block(net, from_layer, num_filter, block_id, bottleneck_fact=0.5, stride=2, pad=1, use_bn=True):

  branch1 = '{}'.format(block_id)
  ConvBNLayer(net, from_layer, branch1, use_bn=use_bn, use_relu=False, num_output=num_filter, kernel_size=1, pad=0, stride=stride)


  branch2a = '{}/b2a'.format(block_id)
  ConvBNLayer(net, from_layer, branch2a, use_bn=use_bn, use_relu=True, num_output=int(num_filter*bottleneck_fact), kernel_size=1, pad=0, stride=1)

  branch2b = '{}/b2b'.format(block_id)
  ConvBNLayer(net, branch2a, branch2b, use_bn=use_bn, use_relu=True, num_output=int(num_filter*bottleneck_fact), kernel_size=3, pad=pad, stride=stride)

  branch2c = '{}/b2c'.format(block_id)
  ConvBNLayer(net, branch2b, branch2c, use_bn=use_bn, use_relu=False, num_output=num_filter, kernel_size=1, pad=0, stride=1)

  res_name = '{}/res'.format(block_id)
  net[res_name] = L.Eltwise(net[branch1], net[branch2c])
  relu_name = '{}/relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

  return relu_name


def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, dilation=1, use_scale=True, lr_mult=1,
    conv_prefix='', conv_postfix='', bn_prefix='', bn_postfix='/bn',
    scale_prefix='', scale_postfix='/scale', bias_prefix='', bias_postfix='/bias',
    **bn_params):
  if use_bn:
    # parameters for convolution layer with batchnorm. weight_filler=dict(type='xavier'), bias_filler=dict(type='constant')
    kwargs = {
        'param': [dict(lr_mult=lr_mult, decay_mult=1)],
        'weight_filler': dict(type='xavier'),
        'bias_term': False,
        }
    eps = bn_params.get('eps', 0.001)
    moving_average_fraction = bn_params.get('moving_average_fraction', 0.999)
    #moving_average_fraction = bn_params.get('moving_average_fraction', 0.1)
    use_global_stats = bn_params.get('use_global_stats', False)
    # parameters for batchnorm layer.
    bn_kwargs = {
        'param': [
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0),
            dict(lr_mult=0, decay_mult=0)],
        'eps': eps,
        'moving_average_fraction': moving_average_fraction,
        }
    bn_lr_mult = lr_mult
    if use_global_stats:
      # only specify if use_global_stats is explicitly provided;
      # otherwise, use_global_stats_ = this->phase_ == TEST;
      bn_kwargs = {
          'param': [
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0),
              dict(lr_mult=0, decay_mult=0)],
          'eps': eps,
          'use_global_stats': use_global_stats,
          }
      # not updating scale/bias parameters
      bn_lr_mult = 0
    # parameters for scale bias layer after batchnorm.
    if use_scale:
      sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
          }
    else:
      bias_kwargs = {
          'param': [dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=0.0),
          }
  else:
    kwargs = {
        'param': [
            dict(lr_mult=lr_mult, decay_mult=1),
            dict(lr_mult=2 * lr_mult, decay_mult=0)],
        'weight_filler': dict(type='xavier'),
        'bias_filler': dict(type='constant', value=0)
        }

  conv_name = '{}{}{}'.format(conv_prefix, out_layer, conv_postfix)
  [kernel_h, kernel_w] = UnpackVariable(kernel_size, 2)
  [pad_h, pad_w] = UnpackVariable(pad, 2)
  [stride_h, stride_w] = UnpackVariable(stride, 2)
  if kernel_h == kernel_w:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_size=kernel_h, pad=pad_h, stride=stride_h, **kwargs)
  else:
    net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,
        kernel_h=kernel_h, kernel_w=kernel_w, pad_h=pad_h, pad_w=pad_w,
        stride_h=stride_h, stride_w=stride_w, **kwargs)
  if dilation > 1:
    net.update(conv_name, {'dilation': dilation})
  if use_bn:
    bn_name = '{}{}{}'.format(bn_prefix, out_layer, bn_postfix)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, **bn_kwargs)
    if use_scale:
      sb_name = '{}{}{}'.format(scale_prefix, out_layer, scale_postfix)
      net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
    else:
      bias_name = '{}{}{}'.format(bias_prefix, out_layer, bias_postfix)
      net[bias_name] = L.Bias(net[bn_name], in_place=True, **bias_kwargs)
  if use_relu:
    relu_name = '{}/relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)



def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret
