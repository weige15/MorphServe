import torch

from .kernels.block_mgmt import set_block_table_and_num_seq_alloc_blocks, unset_block_table_and_num_seq_alloc_blocks, gather_allocated_blocks_and_unset

class BlockManager:
    """
    BlockManager - Manage the block table and free blocks on CPU / GPU

    This manager records the mapping from (sequence ID, block index) to block 
    ID (which we call `block_table`), and provides methods to allocate and free
    blocks.

    All tables (the block table, the `num_seq_allocated_blocks`, and the free block
    list) are all maintained on the GPU, so that we can leverage custom Triton
    kernels for fast operations.
    """

    def __init__(self, device_name: str, num_blocks: int, max_seqs_in_block_table: int, max_blocks_per_seq: int, block_size: int):
        self.device_name = device_name
        self.num_free_blocks = num_blocks
        self.num_blocks = num_blocks
        self.block_size = block_size

        # seq_id |-> number of blocks allocated for this sequence
        self.num_seq_allocated_blocks = torch.zeros(
            (max_seqs_in_block_table,),
            dtype=torch.int32,
            device="cuda"
        )
        # (seq_id, block_index) |-> block_id
        self.block_table = torch.empty(
            (max_seqs_in_block_table, max_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
        # block_id |-> whether this block is free or not
        self.is_block_free = torch.ones(
            (num_blocks,),
            dtype=torch.bool,
            device="cuda"
        )
    
    def _allocate_blocks(self, num_blocks: int) -> torch.Tensor:
        """
        Allocate the requested number of blocks, update relevant status, and
        return the block IDs.
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(f"No enough free blocks available on {self.device_name} ({self.num_blocks} in total, {self.num_free_blocks} free, {num_blocks} requested)")
        selected_blocks = torch.nonzero(self.is_block_free)[:num_blocks].view(-1)
        self.num_free_blocks -= num_blocks
        self.is_block_free[selected_blocks] = False
        return selected_blocks
    
    def _free_blocks(self, block_ids: torch.Tensor):
        """
        Free the specified blocks, and update relevant status.
        """
        self.num_free_blocks += len(block_ids)
        self.is_block_free[block_ids] = True
    
    def allocate_blocks_for_seqs(self, seq_ids: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """
        Allocate blocks for sequences, making sure that seq #i has at least 
        ceil(target_lengths[i] / block_size) blocks allocated.

        Return new blocks allocated for the sequences. (useful for swapping)
        """
        target_num_blocks = (target_lens + (self.block_size-1)) // self.block_size
        assert (self.num_seq_allocated_blocks[seq_ids] <= target_num_blocks).all(), \
            f"""(On {self.device_name}) Logic error: Some sequences have more blocks already allocated than needed.
                seq_ids: {seq_ids}, target_lens: {target_lens}, target_num_blocks: {target_num_blocks},
                self.num_seq_allocated_blocks[seq_ids]: {self.num_seq_allocated_blocks[seq_ids]}"""
        block_needed = target_num_blocks - self.num_seq_allocated_blocks[seq_ids]
        new_blocks = self._allocate_blocks(torch.sum(block_needed).item())

        set_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, new_blocks, seq_ids, block_needed)

        return new_blocks
        
    def free_blocks_for_seqs(self, seq_ids: torch.Tensor):
        """
        Free blocks for sequences.
        """
        self.num_free_blocks += torch.sum(self.num_seq_allocated_blocks[seq_ids]).item()
        unset_block_table_and_num_seq_alloc_blocks(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)

    def gather_allocated_blocks_and_free(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Gather the block IDs allocated for the specified sequences and mark them as free

        Useful fow swapping in/out
        """
        gathered_block_ids = gather_allocated_blocks_and_unset(self.num_seq_allocated_blocks, self.block_table, seq_ids, self.is_block_free)
        self.num_free_blocks += len(gathered_block_ids)
        return gathered_block_ids

    def get_num_allocated_blocks(self, seq_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the number of blocks allocated for the specified sequences
        Useful for swapping
        """
        return self.num_seq_allocated_blocks[seq_ids]

    def get_allocated_block_ids(self) -> torch.Tensor:
        """Return every currently allocated virtual block ID."""
        return torch.nonzero(~self.is_block_free, as_tuple=False).view(-1)

    def extend(self, additional_blocks: int):
        """Append physically-backed virtual IDs without changing existing IDs."""
        if additional_blocks <= 0:
            raise ValueError("additional_blocks must be positive")
        self.is_block_free = torch.cat((
            self.is_block_free,
            torch.ones(additional_blocks, dtype=torch.bool, device=self.is_block_free.device),
        ))
        self.num_blocks += additional_blocks
        self.num_free_blocks += additional_blocks
        self.assert_consistent()

    def plan_compaction_to_prefix(
        self,
        prefix_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose extension sources and free base destinations without mutation."""
        if not 0 < prefix_blocks <= self.num_blocks:
            raise ValueError("invalid compaction prefix")
        allocated = self.num_blocks - self.num_free_blocks
        if allocated > prefix_blocks:
            raise RuntimeError(
                f"cannot compact {allocated} allocated blocks into {prefix_blocks}"
            )
        extension_ids = (
            torch.nonzero(~self.is_block_free[prefix_blocks:], as_tuple=False).view(-1)
            + prefix_blocks
        )
        if extension_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=self.is_block_free.device)
            return empty, empty
        base_ids = torch.nonzero(
            self.is_block_free[:prefix_blocks], as_tuple=False
        ).view(-1)[:extension_ids.numel()]
        if base_ids.numel() != extension_ids.numel():
            raise RuntimeError("not enough free base blocks for compaction")
        return extension_ids, base_ids

    def commit_compaction(
        self,
        extension_ids: torch.Tensor,
        base_ids: torch.Tensor,
    ):
        """Publish a completed physical extension-to-base copy in the block table."""
        if extension_ids.numel() != base_ids.numel():
            raise ValueError("compaction source/destination counts differ")
        if extension_ids.numel() == 0:
            return
        if not bool((~self.is_block_free[extension_ids]).all().item()):
            raise RuntimeError("compaction source is not allocated")
        if not bool(self.is_block_free[base_ids].all().item()):
            raise RuntimeError("compaction destination is not free")

        id_map = torch.arange(
            self.num_blocks, dtype=torch.int64, device=self.is_block_free.device
        )
        id_map[extension_ids] = base_ids
        allocated_counts = self.num_seq_allocated_blocks.cpu().tolist()
        for seq_id, count in enumerate(allocated_counts):
            if count:
                row = self.block_table[seq_id, :count]
                self.block_table[seq_id, :count] = id_map[row.long()].to(torch.int32)

        self.is_block_free[base_ids] = False
        self.is_block_free[extension_ids] = True
        self.assert_consistent()

    def shrink(self, num_blocks: int):
        """Remove a free virtual-ID suffix after physical compaction."""
        if not 0 < num_blocks <= self.num_blocks:
            raise ValueError("invalid shrink target")
        if not bool(self.is_block_free[num_blocks:].all().item()):
            raise RuntimeError("cannot shrink while suffix blocks are allocated")
        self.is_block_free = self.is_block_free[:num_blocks].clone()
        self.num_blocks = num_blocks
        self.num_free_blocks = int(self.is_block_free.sum().item())
        self.assert_consistent()

    def assert_consistent(self):
        observed_free = int(self.is_block_free.sum().item())
        if tuple(self.is_block_free.shape) != (self.num_blocks,):
            raise RuntimeError("free bitmap length does not match num_blocks")
        if observed_free != self.num_free_blocks:
            raise RuntimeError(
                f"free block count mismatch: {self.num_free_blocks} != {observed_free}"
            )
        allocated_from_rows = int(self.num_seq_allocated_blocks.sum().item())
        if allocated_from_rows != self.num_blocks - self.num_free_blocks:
            raise RuntimeError(
                "block-table allocation count does not match free bitmap: "
                f"{allocated_from_rows} != {self.num_blocks - self.num_free_blocks}"
            )
    