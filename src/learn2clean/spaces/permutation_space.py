import math
from typing import TypeVar

import numpy as np
from gymnasium import spaces

T = TypeVar("T")


class PermutationSpace(spaces.Discrete):
    """
    An implicit Gymnasium Space representing all ordered partial permutations
    of a set of items.

    Unlike a standard permutation space that only considers full sequences of length N,
    this space covers all possible ordered sequences of length 1 to N (inclusive).

    The space maps a unique integer index to a specific sequence using a combinatorial
    unranking algorithm (Rank/Unrank). This avoids storing the factorial number of
    possibilities in memory.

    Mathematical Definition:
        Space Size = Sum(P(N, k)) for k in [1, N]

    Attributes:
        items (list[T]): The list of atomic items available.
        n_items (int): The number of atomic items.
        n (int): The total cardinality of the space.
    """

    def __init__(self, items: list[T], seed: int | None = None):
        """
        Initialize the PermutationSpace.

        Args:
            items: A list of distinct items (actions, strings, objects).
            seed: An optional seed for the random number generator (RNG).

        Raises:
            ValueError: If the item list is empty or exceeds 20 items (int64 overflow).
        """
        if not items:
            raise ValueError("PermutationSpace requires at least one item.")

        self.items = items
        self.n_items = len(items)

        # Safety Check: 20! fits in int64, 21! does not.
        # Note: Even 10! (3.6M) is extremely large for standard RL agents.
        if self.n_items > 20:
            raise ValueError(
                f"PermutationSpace overflow: {self.n_items} items result in a space "
                "size exceeding int64 capacity. Max supported is 20."
            )

        # Pre-calculate the number of permutations for each length bucket k (1 to N)
        # bucket_counts[i] stores P(N, i+1)
        self.bucket_counts = [
            math.perm(self.n_items, r) for r in range(1, self.n_items + 1)
        ]

        # Total cardinality is the sum of all k-permutations
        total_permutations = sum(self.bucket_counts)

        super().__init__(n=total_permutations, seed=seed)

    def idx_to_permutation(self, idx: int | np.integer) -> tuple[T, ...]:
        """
        Decode a unique integer index into its corresponding sequence of items.

        This method reconstructs the k-permutation in O(k) time without generating
        any other permutation, making it memory efficient.

        Args:
            idx: The integer index to decode.

        Returns:
            tuple[T, ...]: The ordered sequence of items.

        Raises:
            IndexError: If the index is outside the valid range [0, self.n).
        """
        # Ensure native python int for large number arithmetic
        idx = int(idx)

        if not 0 <= idx < self.n:
            raise IndexError(f"Index {idx} out of bounds (Max: {self.n})")

        # --- Step 1: Determine the length of the target permutation ---
        # We iterate through the 'buckets' of lengths (length 1, length 2, etc.)
        target_length = 0
        current_idx = idx

        for r, count in enumerate(self.bucket_counts, start=1):
            if current_idx < count:
                target_length = r
                break
            current_idx -= count

        # --- Step 2: Unrank the specific permutation of length 'target_length' ---
        # Logic based on the Factoradic Number System variant for k-permutations.
        available = list(self.items)
        perm = []

        for i in range(target_length):
            # Calculate the "block size": how many combinations exist for the
            # REMAINING items if we fix the current position.
            # Block Size = P(n_remaining - 1, length_needed_after_this_one)
            n_remaining = len(available)
            items_needed_after = target_length - 1 - i

            block_size = math.perm(n_remaining - 1, items_needed_after)

            # Determine which item index to pick
            selected_idx = current_idx // block_size

            # Add to permutation and remove from available pool
            perm.append(available.pop(selected_idx))

            # Update index to point within the selected block (modulo)
            current_idx %= block_size

        return tuple(perm)
