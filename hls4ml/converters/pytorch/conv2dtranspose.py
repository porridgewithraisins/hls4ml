from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import compute_padding_2d_pytorch, parse_data_format


@pytorch_handler('ConvTranspose2d')
def parse_conv2dtranspose_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    """Handles PyTorch's nn.ConvTranspose2d layer.

    Args:
        operation: String identifying the layer type ('ConvTranspose2d')
        layer_name: Name of this layer in the model
        input_names: List of names of the input tensors
        input_shapes: List of shapes of the input tensors
        node: Node from the PyTorch FX graph
        class_object: The PyTorch layer object (nn.ConvTranspose2d instance)
        data_reader: Object to read layer weights
        config: Dictionary of configuration parameters

    Returns:
        (layer, output_shape): Tuple of the layer parameters and output shape
    """
    assert 'ConvTranspose2d' in operation

    layer = {}

    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['class_name'] = 'Conv2DTranspose'
    layer['data_format'] = 'channels_last'  # We'll convert from PyTorch's channels_first

    # Get weights and biases - need to transpose from PyTorch's format
    weights = class_object.weight.data.numpy()
    # PyTorch Conv2DTranspose weight format: (in_channels, out_channels, height, width)
    # We need: (height, width, in_channels, out_channels)
    layer['weight_data'] = weights.transpose(2, 3, 0, 1)

    if class_object.bias is not None:
        layer['bias_data'] = class_object.bias.data.numpy()
    else:
        layer['bias_data'] = None

    # Input info - converting from channels_first to channels_last
    (*_, in_height, in_width, layer['n_chan']) = parse_data_format(input_shapes[0], 'channels_first')
    layer['in_height'] = in_height
    layer['in_width'] = in_width

    # Layer parameters
    layer['n_filt'] = class_object.out_channels
    layer['filt_height'] = class_object.kernel_size[0]
    layer['filt_width'] = class_object.kernel_size[1]
    layer['stride_height'] = class_object.stride[0]
    layer['stride_width'] = class_object.stride[1]

    # Calculate output dimensions
    if isinstance(class_object.padding, tuple):
        layer['pad_top'] = layer['pad_bottom'] = class_object.padding[0]
        layer['pad_left'] = layer['pad_right'] = class_object.padding[1]
    else:
        layer['pad_top'] = layer['pad_bottom'] = class_object.padding
        layer['pad_left'] = layer['pad_right'] = class_object.padding

    # PyTorch calculates transposed conv output as:
    # H_out = (H_in - 1) * stride + kernel - 2 * padding
    layer['out_height'] = (layer['in_height'] - 1) * layer['stride_height'] + layer['filt_height'] - 2 * layer['pad_top']
    layer['out_width'] = (layer['in_width'] - 1) * layer['stride_width'] + layer['filt_width'] - 2 * layer['pad_left']

    # Pytorch uses channels_first, but we're converting to channels_last for output
    output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]

    return layer, output_shape
