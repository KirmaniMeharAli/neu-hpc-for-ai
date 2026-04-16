"""Part 5 (stretch): Paged KV Cache

Fixed-size blocks instead of contiguous allocation.
Same idea as OS virtual memory pages.
"""

import math
import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers

        self.k_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self.free_blocks: list[int] = list(range(num_blocks))
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence. Returns list of block IDs."""
        num_blocks_needed = math.ceil(num_tokens / self.block_size)

        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        # Pop blocks from the free list
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)

        # Track which blocks belong to this sequence
        if seq_id not in self.seq_to_blocks:
            self.seq_to_blocks[seq_id] = []
        self.seq_to_blocks[seq_id].extend(allocated)

        return allocated

    def free(self, seq_id: int):
        """Free all blocks for a finished sequence."""
        if seq_id not in self.seq_to_blocks:
            return

        blocks = self.seq_to_blocks.pop(seq_id)

        # Zero out the block memory for each layer
        for block_id in blocks:
            for layer_idx in range(self.num_layers):
                self.k_pool[layer_idx][block_id].zero_()
                self.v_pool[layer_idx][block_id].zero_()

        # Return blocks to the free list
        self.free_blocks.extend(blocks)

    def get_block_ids(self, seq_id: int) -> list[int]:
        return self.seq_to_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
