import torch
import torch.nn as nn



class ConvLSTMCell(nn.Module):
    """
    The basic ConvLSTMCell unit without implementing dropout of hidden and input conv
    and layer norm
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 bias):
        """
            Initialize ConvLSTM cell.

            Parameters
            ----------
            input_dim: int
                Number of channels of input tensor.
            hidden_dim: int
                Number of channels of hidden state.
            kernel_size: (int, int)
                Size of the convolutional kernel.
            bias: bool
                Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.in_channels + self.hidden_channels,
                              out_channels=self.hidden_channels * 4,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, state):
        hidden, cell = state
        combined = torch.cat([input_tensor, hidden], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        cell = f * cell + i * g
        hidden = o * torch.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 num_layers=1,
                 batch_first=False,
                 bias=True,
                 bidirectional=False,
                 return_all_layers=False):

        super(ConvLSTM, self).__init__()
        self.check_kernel_size_consistency(kernel_size)

        kernel_size = self.extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self.extend_for_multilayer(hidden_channels, num_layers)

        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.bidirectional = bidirectional
        self.return_all_layers = return_all_layers

        # The forward pass cells
        cells_forward = []
        for i in range(self.num_layers):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i-1]
            cells_forward.append(
                ConvLSTMCell(in_channels=cur_in_channels,
                             hidden_channels=self.hidden_channels[i],
                             kernel_size=self.kernel_size[i],
                             bias=self.bias)
            )
        self.cells_forward = nn.ModuleList(cells_forward)

        # The backward pass cells if bidirectional
        if self.bidirectional:
            cells_backward = []
            for i in range(self.num_layers):
                cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i - 1]
                cells_backward.append(
                    ConvLSTMCell(
                        in_channels=cur_in_channels,
                        hidden_channels=self.hidden_channels[i],
                        kernel_size=self.kernel_size[i],
                        bias=self.bias
                    )
                )
            self.cells_backward = nn.ModuleList(cells_backward)

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: 5 D tensor for video data
        :param hidden_state: TODO: implement stateful
        :return: (output, last_hidden_state)
        """
        if not self.batch_first:
            input_tensor = torch.transpose(input_tensor, 0, 1)

        b, seq_len, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError
        else:
            states_forward, states_backward = self.init_hidden(b, [h, w])

        # Some default values and instantiation
        layer_outputs_forward_list = []
        h_fw, c_fw, h_bw, c_bw = None, None, None, None
        last_state_backward = None

        # Forward direction
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h_fw, c_fw = states_forward[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_fw, c_fw = self.cells_forward[layer_idx](input_tensor=cur_layer_input[:, t],
                                                     state=[h_fw, c_fw])
                output_inner.append(h_fw)
            # layer_output.shape = [b,t,c,h,w]
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_outputs_forward_list.append(layer_output)

        # 6D tensor, layer_outputs.shape = [b,l,t,c,h,w]
        layer_outputs = torch.stack(layer_outputs_forward_list, dim=1)
        last_state_forward = [h_fw, c_fw]

        # Backward direction if bidirectional
        if self.bidirectional:
            cur_layer_input_inv = input_tensor
            layer_outputs_backward_list = []
            for layer_idx in range(self.num_layers):
                h_bw, c_bw = states_backward[layer_idx]
                output_inner = []
                for t in reversed(range(seq_len)):
                    h_bw, c_bw = self.cells_backward[layer_idx](input_tensor=cur_layer_input_inv[:, t],
                                                                state=[h_bw, c_bw])
                    output_inner.append(h_bw)
                output_inner.reverse()
                layer_output_inv = torch.stack(output_inner, dim=1)
                cur_layer_input_inv = layer_output_inv
                # for every element in the list:
                # [b,t,c,h,w]
                layer_outputs_backward_list.append(layer_output_inv)
            # [b,l,t,c*D,h,w] where D is Direction
            layer_outputs = torch.stack([torch.cat([layer_outputs_forward_list[i], layer_outputs_backward_list[i]], dim=2)
                                         for i in range(self.num_layers)], dim=1)
            last_state_backward = [h_bw, c_bw]

        if self.return_all_layers:
            return layer_outputs, last_state_forward, last_state_backward
        else:
            # return the states of the last layer [b, t, D*c, h, w]
            return layer_outputs[:,-1], last_state_forward, last_state_backward

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        init_hidden_forward = [cell.init_hidden(batch_size, [height, width]) for cell in self.cells_forward]
        init_hidden_backward = None
        if self.bidirectional:
            init_hidden_backward = [cell.init_hidden(batch_size, [height, width]) for cell in self.cells_backward]
        return init_hidden_forward, init_hidden_backward

    @staticmethod
    def check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    test = ConvLSTM(1, 32, (3, 3), 2, True, bidirectional=True)
    input_tensor = torch.zeros(10, 2, 1, 64, 64)
    output,_,_ = test(input_tensor)
    print(output.shape)