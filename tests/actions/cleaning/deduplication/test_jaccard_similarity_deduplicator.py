from learn2clean.actions import JaccardSimilarityDeduplicator
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_js_deduplicator_identity():
    """
    Tests for perfect similarity (Jaccard = 1.0) and default removal (keep='first').

    P (Tokens): {'quick', 'brown', 'fox'}
    Q (Tokens): {'quick', 'brown', 'fox'}
    Jaccard: 3/3 = 1.0
    """
    assert_action_pipeline(
        JaccardSimilarityDeduplicator(
            columns=["Text"]
        ),  # High threshold guarantees identity check
        {
            "Text": ["quick brown fox", "jumped over", "quick brown fox"],
            "ID": [1, 2, 3],
        },
        {
            # Row 2 (index 2) is dropped in favor of Row 0 (index 0)
            "Text": ["quick brown fox", "jumped over"],
            "ID": [1, 2],
        },
    )


def test_jaccard_deduplication_basic():
    data = {
        "text": [
            "apple banana orange",
            "apple banana",  # High similarity
            "strawberry kiwi",  # Distinct
            "orange banana apple",  # Same words, different order -> Jaccard is 1.0
        ]
    }
    assert_action_pipeline(
        JaccardSimilarityDeduplicator(threshold=0.5, keep="first"),
        data,
        {
            "text": [
                "apple banana orange",
                "strawberry kiwi",
            ]
        },
    )


def test_jaccard_deduplication_precision():
    data = {
        "text": [
            "apple banana orange",
            "apple banana",  # High similarity
            "strawberry kiwi",  # Distinct
            "orange banana apple",  # Same words, different order -> Jaccard is 1.0
        ]
    }
    assert_action_pipeline(
        JaccardSimilarityDeduplicator(threshold=0.7, num_perm=256, keep="first"),
        data,
        {
            "text": [
                "apple banana orange",
                "strawberry kiwi",
            ]
        },
    )


def test_js_deduplication_threshold_sensitivity():
    """
    Tests removal sensitivity by using a pair with Jaccard similarity of 0.20.
    P: {'A', 'B', 'C'}, Q: {'C', 'D', 'E'} -> Jaccard = 1/5 = 0.20
    """
    data = {
        "Key": ["A B C", "X Y Z", "C D E"],
        "Value": [10, 20, 30],
    }

    # Case A: Threshold is 0.3 (Should FAIL to drop, as 0.2 < 0.3)
    assert_action_pipeline(
        JaccardSimilarityDeduplicator(threshold=0.3),
        data,
        data,  # Expected: No rows dropped
    )

    # Case B: Threshold is 0.1 (Should SUCCEED and drop Row 2, as 0.2 >= 0.1)
    assert_action_pipeline(
        JaccardSimilarityDeduplicator(threshold=0.1),
        data,
        {
            "Key": ["A B C", "X Y Z"],
            "Value": [10, 20],
        },
    )
