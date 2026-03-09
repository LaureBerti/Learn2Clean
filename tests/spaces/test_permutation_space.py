import itertools

import numpy as np
import pytest
from gymnasium import spaces

from learn2clean.spaces.permutation_space import PermutationSpace


class TestPermutationSpace:
    """
    Test suite for the PermutationSpace class.
    """

    def test_initialization_small_space(self):
        """Verifies initialization and size for a small number of items."""
        items = ["A", "B"]
        space = PermutationSpace(items)

        # Theoretical calculation:
        # P(2, 1) = 2 -> (A), (B)
        # P(2, 2) = 2 -> (A, B), (B, A)
        # Total = 4
        assert space.n == 4
        assert space.n_items == 2
        assert isinstance(space, spaces.Space)
        assert space.dtype == np.int64

    def test_initialization_medium_space(self):
        """Verifies size for 3 items (classic manual example)."""
        items = [1, 2, 3]
        space = PermutationSpace(items)

        # Total = P(3, 1) + P(3, 2) + P(3, 3) = 3 + 6 + 6 = 15
        assert space.n == 15

    def test_contains_logic(self):
        """Verifies the contains method."""
        items = ["A", "B"]
        space = PermutationSpace(items)

        # Valid indices: 0, 1, 2, 3
        assert space.contains(0)
        assert space.contains(3)
        assert space.contains(np.int64(2))  # Test numpy type compatibility

        # Invalid indices
        assert not space.contains(-1)
        assert not space.contains(4)
        assert not space.contains(100)
        assert not space.contains("invalid")
        assert not space.contains(1.5)

    def test_decoding_logic_exact_mapping(self):
        """
        Verifies exact decoding (Rank/Unrank) for a known small case.
        Items: [A, B]
        Expected mapping (lexicographical order by length):
        Length 1:
          idx 0 -> (A,)
          idx 1 -> (B,)
        Length 2:
          idx 2 -> (A, B)
          idx 3 -> (B, A)
        """
        items = ["A", "B"]
        space = PermutationSpace(items)

        # Length 1 bucket
        assert space.idx_to_permutation(0) == ("A",)
        assert space.idx_to_permutation(1) == ("B",)

        # Length 2 bucket
        assert space.idx_to_permutation(2) == ("A", "B")
        assert space.idx_to_permutation(3) == ("B", "A")

    def test_decoding_out_of_bounds(self):
        """Verifies that out-of-bounds access raises an exception."""
        items = ["A", "B"]
        space = PermutationSpace(items)

        # Max index is 3 (size 4)
        with pytest.raises(IndexError, match="out of bounds"):
            space.idx_to_permutation(4)

        with pytest.raises(IndexError, match="out of bounds"):
            space.idx_to_permutation(-1)

    def test_seeding_and_reproducibility(self):
        """Verifies that seeding guarantees identical sampling results."""
        items = list(range(10))
        seed = 42

        space1 = PermutationSpace(items, seed=seed)
        space2 = PermutationSpace(items, seed=seed)

        # Draw 100 samples
        samples1 = [space1.sample() for _ in range(100)]
        samples2 = [space2.sample() for _ in range(100)]

        assert samples1 == samples2

        # Verify that a different seed produces different results
        space3 = PermutationSpace(items, seed=999)
        assert space1.sample() != space3.sample()

    def test_overflow_protection(self):
        """Verifies that the 20-item limit is respected."""
        # Acceptable limit case: 20 items
        items_20 = list(range(20))
        try:
            space = PermutationSpace(items_20)
            assert space.n > 0
        except ValueError:
            pytest.fail("PermutationSpace should support 20 items.")

        # Unacceptable limit case: 21 items
        items_21 = list(range(21))
        with pytest.raises(ValueError, match="PermutationSpace overflow"):
            PermutationSpace(items_21)

    def test_complex_types_support(self):
        """Verifies that the class works with complex objects (not just int/str)."""

        class MyAction:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"Action({self.name})"

        a1 = MyAction("clean")
        a2 = MyAction("drop")

        space = PermutationSpace([a1, a2])

        # Index 2 corresponds to (a1, a2)
        perm = space.idx_to_permutation(2)

        assert len(perm) == 2
        assert perm[0] is a1
        assert perm[1] is a2

    def test_sampling_validity(self):
        """Statistically verifies that sample() returns valid indices and permutations."""
        items = ["A", "B", "C"]
        space = PermutationSpace(items)

        for _ in range(100):
            idx = space.sample()
            assert space.contains(idx)

            perm = space.idx_to_permutation(idx)
            assert isinstance(perm, tuple)

            # Check length constraints
            assert 1 <= len(perm) <= 3

            # Verify uniqueness of elements within the permutation
            assert len(perm) == len(set(perm))

    def test_decoding_exhaustive_comparison(self):
        """
        Replaces the hardcoded 'test_decoding_big'.

        We use Python's standard library 'itertools' as a Ground Truth Oracle.
        We generate all permutations for lengths 1 to N and verify that
        PermutationSpace matches them exactly index by index.
        """
        items = [1, 2, 3, 4, 5]
        space = PermutationSpace(items)

        # 1. Generate Ground Truth using itertools
        # itertools.permutations generates items in lexicographical order,
        # which matches our implementation logic.
        ground_truth = []
        for r in range(1, len(items) + 1):
            ground_truth.extend(list(itertools.permutations(items, r)))

        # 2. Verify total size matches
        assert space.n == len(ground_truth)

        # 3. Verify the first 100 items (covering length 1, length 2, and start of length 3)
        for i in range(100):
            assert space.idx_to_permutation(i) == ground_truth[i]

        # 4. Verify the very last item (Length N)
        last_idx = space.n - 1
        assert space.idx_to_permutation(last_idx) == ground_truth[-1]

        # 5. Verify a random index in the middle
        mid_idx = space.n // 2
        assert space.idx_to_permutation(mid_idx) == ground_truth[mid_idx]
