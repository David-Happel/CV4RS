def output_size(d_n, h_n, w_n, kernel_n, padding, stride = 1, dilation = 0): 
    
    d_out_len =  ((d_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    h_out_len =  ((h_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    w_out_len =  ((w_n + (2 * padding) - dilation * (kernel_n - 1) - 1) / stride) + 1
    return d_out_len, h_out_len, w_out_len


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()