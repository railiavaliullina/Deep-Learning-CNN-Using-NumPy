from netlib.Layer import BaseLayerClass
from configs.config import cfg
import numpy as np


def get_indices(x_shape, kernel_height, kernel_width, padding=1, stride=1, c_=None):
    """
    :param x_shape (tuple): shape of input tensor
    :param kernel_height (int): filter size
    :param kernel_width (int): filter size
    :param padding (int)
    :param stride (int)
    :return: k, i, j - indices
    """
    h, w = x_shape[2], x_shape[3]
    c = c_ if c_ is not None else x_shape[1]
    output_h = (h + 2 * padding - kernel_height) // stride + 1
    output_w = (w + 2 * padding - kernel_width) // stride + 1

    k = np.repeat(np.arange(c), kernel_height * kernel_width).reshape(-1, 1)

    i1 = np.tile(np.repeat(np.arange(kernel_height), kernel_width), c).reshape(-1, 1)
    i2 = (np.repeat(np.arange(output_h), output_w) * stride).reshape(1, -1)
    i = np.sum((i1, i2))

    j1 = np.tile(np.arange(kernel_width), kernel_height * c).reshape(-1, 1)
    j2 = (np.tile(np.arange(output_w), output_h) * stride).reshape(1, -1)
    j = np.sum((j1, j2))
    return k, i, j


def tensor2matrix(x, kernel_height, kernel_width, padding=1, stride=1, indices=None):  # conv, max_pool forward
    """
    :param x: input tensor
    :param kernel_height (int): filter size
    :param kernel_width (int): filter size
    :param padding (int)
    :param stride (int)
    :return: converted matrix
    """
    # get_indices(x, k_width, k_height, pad, stride);
    # apply indices to tensor: x_cols = x[:, k, i, j] (for previous example x_cols.shape (1, 8, 4));
    # (batch_size, x_depth*k_width*k_height, output_width*output_height)
    # reshape new tensor: result = x_cols.transpose(1, 2, 0).reshape(k_width*k_height*x_depth, -1)
    c = x.shape[1]
    k, i, j = get_indices(x.shape, kernel_height, kernel_width, padding, stride) if indices is None else indices
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    converted_x = x[:, k, i, j]
    converted_x = converted_x.transpose(1, 2, 0).reshape(kernel_height * kernel_width * c, -1)
    return converted_x


def matrix2tensor(grad_input, input_shape, kernel_height, kernel_width, padding=1, stride=1):  # conv, max_pool backward
    """
    :param grad_input: input tensor of gradients
    :param input_shape (tuple): shape of input tensor
    :param kernel_height (int): filter size
    :param kernel_width (int): filter size
    :param padding (int)
    :param stride (int)
    :return: converted matrix
    """
    b_size, c, h, w = input_shape
    # Initialize new tensor with zeros(input.shape);
    # get indices k, i, j (as in forward pass);
    # reshape input tensor to size=(k_depth * k_height * k_width, output_width*output_height, batch_size);
    # transpose reshaped tensor to size=(batch_size, k_depth * k_height * k_width, output_width*output_height);
    # apply indices to sum values on the same positions (use np.add.at);
    # remove padded part.
    converted_tensor = np.zeros((b_size, c, h + 2 * padding, w + 2 * padding))
    k, i, j = get_indices(input_shape, kernel_height, kernel_width, padding, stride)
    reshaped_input = grad_input.reshape(c * kernel_height * kernel_width, -1, b_size)
    transposed_reshaped_input = reshaped_input.transpose(2, 0, 1)
    np.add.at(converted_tensor, (slice(None), k, i, j), transposed_reshaped_input)
    converted_tensor = converted_tensor[:, :, padding:-padding, padding:-padding] if padding != 0 else converted_tensor
    return converted_tensor


class Convolution(BaseLayerClass):
    def __init__(self, kernel_size, nrof_filters, kernel_depth, zero_pad, stride, use_bias, initialization_type):
        self.cfg = cfg
        self.kernel_size = kernel_size
        self.nrof_filters = nrof_filters
        self.kernel_depth = kernel_depth
        self.zero_pad = zero_pad
        self.stride = stride
        self.use_bias = use_bias
        self.initialization_type = initialization_type

        self.init_weights()

    def trainable(self):
        return True

    def init_weights(self):
        k_width, k_height = self.kernel_size if isinstance(self.kernel_size, tuple) \
            else (self.kernel_size, self.kernel_size)
        k_depth = self.kernel_depth
        n_filters = self.nrof_filters
        n = k_width * k_height * k_depth
        m = n_filters

        if self.initialization_type.name.endswith('Normal'):
            if self.initialization_type.name == 'Normal':
                scale = self.cfg['convolution_layer']['scale']
            elif self.initialization_type.name == 'HeNormal':
                scale = np.sqrt(2 / n)
            elif self.initialization_type.name == 'XavierNormal':
                scale = np.sqrt(1 / n)
            elif self.initialization_type.name == 'GlorotNormal':
                scale = np.sqrt(1 / (n + m))
            else:
                raise Exception
            self.W = np.random.normal(size=[n_filters, k_depth, k_width, k_height], scale=scale)
            self.b = np.random.normal(size=[n_filters], scale=scale) if self.use_bias else None
        else:
            if self.initialization_type.name == 'Uniform':
                l = self.cfg['convolution_layer']['l']
            elif self.initialization_type.name == 'HeUniform':
                l = np.sqrt(6 / n)
            elif self.initialization_type.name == 'XavierUniform':
                l = np.sqrt(3 / n)
            elif self.initialization_type.name == 'GlorotUniform':
                l = np.sqrt(6 / (n + m))
            else:
                raise Exception
            self.W = np.random.uniform(-l, l, size=[n_filters, k_depth, k_width, k_height])
            self.b = np.random.uniform(-l, l, size=[n_filters]) if self.use_bias else None

    def __call__(self, input, phase='eval'):
        """
        :param input: input tensor, shape=(batch_size, depth, height, width)
        :param phase: ['train', 'eval']
        :return: output of convolutional layer
        """
        weights, stride, padding, bias = self.W, self.stride, self.zero_pad, self.b
        b_size, c, h, w = input.shape
        n_filters, k_depth, k_height, k_width = weights.shape
        output_h = (h + 2 * padding - k_height) // stride + 1
        output_w = (w + 2 * padding - k_width) // stride + 1

        # Convert input tensor to matrix (shape: (x_depth*k_width*k_height, output_width*output_height*batch_size));
        # reshape weights (new shape: (n_filters, depth*k_width*k_height));
        # multiply reshaped weights and converted tensor;
        # reshape output tensor (from (n_filters, output_width*output_height*batch_size) to (n_filters, output_width,
        # output_height, batch_size);
        # transpose: result = reshaped_output.transpose(3, 0, 1, 2).
        converted_input = tensor2matrix(input, k_height, k_width, padding, stride)
        reshaped_weights = weights.reshape((n_filters, -1))
        weights_converted_input_matmul = reshaped_weights @ converted_input
        if bias is not None:
            weights_converted_input_matmul += bias.reshape(-1, 1)

        reshaped_output_tensor = weights_converted_input_matmul.reshape(n_filters, output_h, output_w, b_size)
        transposed_output_tensor = reshaped_output_tensor.transpose(3, 0, 1, 2)

        self.saved_data = {'input': input, 'w': weights, 'b': bias, 'stride': stride, 'padding': padding,
                           'converted_input': converted_input}
        return transposed_output_tensor

    def backward(self, layer_input, dy):
        stride, pad = self.stride, self.zero_pad
        input = self.saved_data['input']
        weights = self.saved_data['w']
        stride = self.saved_data['stride']
        converted_input = self.saved_data['converted_input']
        n_filters,  k_depth, k_height, k_width = weights.shape

        # Reshape input gradients: size = (n_filters, output_w*output_h*batch_size);
        # reshaped_gradients @ converted_tensor.T;
        # reshape output matrix according shape of weights.
        reshaped_input_grads = dy.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        dw = reshaped_input_grads @ converted_input.T
        dw = dw.reshape(weights.shape)
        if self.use_bias:
            db = np.sum(dy, axis=(0, 2, 3))
            dw = (dw, db)
        # reshape weights: size=(n_filters, k_width*k_height*k_depth); (get from forward pass)
        # matmul reshaped weights and reshaped gradients: size=(k_width*k_height*k_depth, output_w*output_h*batch_size);
        # convert matrix to tensor.
        reshaped_weights = weights.reshape(n_filters, -1)
        weights_input_grads_matmul = reshaped_weights.T @ reshaped_input_grads
        dx = matrix2tensor(weights_input_grads_matmul, input.shape, k_height, k_width, pad, stride)
        return dx, dw


class Pooling(BaseLayerClass):
    def __init__(self, kernel_size, stride, type='Max'):
        """
        :param kernel_size (int): kernel size
        :param stride (int): stride
        :param type (string): ['Max', 'Avg']
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type

    def __call__(self, input, phase='eval'):
        """
       :param input: input tensor, shape=(batch_size, depth, height, width)
       :param phase: ['train', 'eval']
       :return: output of pooling layer
       """
        k_width, k_height = self.kernel_size if isinstance(self.kernel_size, tuple) \
            else (self.kernel_size, self.kernel_size)
        b_size, c, h, w = input.shape
        output_h = (h - k_height) // self.stride + 1
        output_w = (w - k_width) // self.stride + 1

        # Reshape input tensor: x.reshape(batch_size*x_depth, 1, x_height, x_width)
        # Convert input tensor (shape: (1*k_width*k_height, output_width*output_height*batch_size*x_depth));
        # apply max/average pooling for each row;
        # reshape output tensor (from (output_width*output_height*batch_size*x_depth) to (output_width,
        # output_height, batch_size, x_depth);
        # transpose: result = reshaped_output.transpose(2, 3, 0, 1).
        x_reshaped = input.reshape(b_size * c, 1, h, w)
        converted_x = tensor2matrix(x_reshaped, k_height, k_width, padding=0, stride=self.stride)

        if self.type == 'Max':
            out = np.max(converted_x, 0)
        elif self.type == 'Avg':
            out = np.mean(converted_x, 0)
        else:
            raise Exception
        out = out.reshape(output_h, output_w, b_size, c)
        out = out.transpose(2, 3, 0, 1)

        self.saved_data = {'input': input, 'converted_input': converted_x}
        return out

    def backward(self, layer_input, dy):
        converted_x = self.saved_data['converted_input']
        input = self.saved_data['input']
        b_size, c, h, w = input.shape
        k_width, k_height = self.kernel_size if isinstance(self.kernel_size, tuple) \
            else (self.kernel_size, self.kernel_size)

        # Initialize tenzor with zeros, shape=(1*k_width*k_height, output_width*output_height*batch_size*x_depth);
        # transpose input tensor of gradients: (x_width, x_height, batch_size, x_depth);
        # view transposed tensor;
        # apply backward max/average-pooling for each row;
        # convert matrix to tensor;
        # reshape from (batch_size*x_depth, 1, x_height, x_width) to (batch_size, x_depth, x_height, x_width).
        zeros_tensor = np.zeros_like(converted_x)
        transposed_input_grads = dy.transpose(2, 3, 0, 1)
        flattened_transposed_input_grads = transposed_input_grads.flatten()
        if self.type == 'Max':
            # backward max-pooling
            max_ids = np.argmax(converted_x, axis=0)
            matrix_coords = (max_ids, np.arange(len(max_ids)))
            zeros_tensor[matrix_coords] = flattened_transposed_input_grads
        elif self.type == 'Avg':
            # backward average-pooling
            zeros_tensor += np.ones(zeros_tensor.shape) * 1/zeros_tensor.shape[0]

        converted_matrix = matrix2tensor(zeros_tensor, (b_size * c, 1, h, w), k_height, k_width, padding=0, stride=self.stride)
        pooling_grad = converted_matrix.reshape(input.shape)
        return pooling_grad, None
