# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import triton
import triton.language as tl


@triton.jit
def memory_gather_kernel(
    table,
    ids,
    output,
    WIDTH: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + row)
    # GPU embedding checks IDs before the entry dependency. Mask additionally
    # so an invalid input cannot cause an out-of-bounds mapped-host read.
    values = tl.load(
        table + token * WIDTH + column,
        mask=(column < WIDTH) & (token >= 0) & (token < VOCAB),
        other=0,
    )
    tl.store(output + row * WIDTH + column, values, mask=column < WIDTH)
