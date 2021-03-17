# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from fairseq import utils
from .conv_tbc import PoincareConvTBC
from fairseq.incremental_decoding_utils import with_incremental_state
from .linear import poincare_linear


@with_incremental_state
class PoincareLinearizedConv1d(PoincareConvTBC):
    """An optimized version of nn.Conv1d.

    At training time, this module uses ConvTBC, which is an optimized version
    of Conv1d. At inference time, it optimizes incremental generation (i.e.,
    one time step at a time) by replacing the convolutions with linear layers.
    Note that the input order changes from training to inference.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = PoincareConvTBC.state_dict(self, destination, prefix, keep_vars=keep_vars)
        # don't store redundant _linearized_weight in checkpoints
        if prefix + '_linearized_weight' in state:
            del state[prefix + '_linearized_weight']
        return state

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        if prefix + '_linearized_weight' in state_dict:
            del state_dict[prefix + '_linearized_weight']

    def forward(self, input, incremental_state=None):
        """
        Args:
            incremental_state: Used to buffer signal; if not None, then input is
                expected to contain a single frame. If the input order changes
                between time steps, call reorder_incremental_state.
        Input:
            Time x Batch x Channel during training
            Batch x Time x Channel during inference
        """
        if incremental_state is None:
            output = super().forward(input)
            if self.kernel_size > 1 and self.padding > 0:
                # remove future timesteps added by padding
                output = output[:-self.padding, :, :]
            return output

        # reshape weight
        kw = self.kernel_size

        bsz = input.size(0)  # input: bsz x len x dim
        if kw > 1:
            input = input.data
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is None:
                input_buffer = input.new(bsz, kw, input.size(2)).zero_()
                self._set_input_buffer(incremental_state, input_buffer)
            else:
                # shift buffer
                input_buffer[:, :-1, :] = input_buffer[:, 1:, :].clone()
            # append next input
            input_buffer[:, -1, :] = input[:, -1, :]
            input = input_buffer
        with torch.no_grad():
            # beta concatenation
            input = self.ball.logmap0(input) * self.beta_n / self.beta_ni
            input = self.ball.expmap0(input.view(bsz, -1))
            output = poincare_linear(
                input, 
                self.weight_g, 
                self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
                self.bias, 
                self.ball.c,
                out_split=1)
        return output.view(bsz, 1, -1)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            input_buffer = input_buffer.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, 'input_buffer')

    def _set_input_buffer(self, incremental_state, new_buffer):
        return utils.set_incremental_state(self, incremental_state, 'input_buffer', new_buffer)
