from learn2clean.actions import ApproximateDeduplicator
from tests.utils.actions.assert_action_pipeline import assert_action_pipeline


def test_approximate_deduplication_default_params() -> None:
    assert_action_pipeline(
        ApproximateDeduplicator(),
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Alice Smith",
                "Bob J",
                "Bob Jones",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "London",
                "Paris",
                "Pariis",
            ],
            "value": [100, 100, 200, 200, 300, 300],
        },
        {
            "id": [1, 2, 3, 5],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Bob J",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "Paris",
            ],
            "value": [100, 100, 200, 300],
        },
    )


def test_approximate_deduplication_transitive():
    """
    Test that transitive duplicates (A~B, B~C) form a single cluster {A, B, C}
    and are handled correctly.
    """
    data = {
        "name": ["John Doe", "Jon Doe", "John Do", "Alice Smith"],
        "id": [1, 2, 3, 4],
    }
    assert_action_pipeline(
        ApproximateDeduplicator(),
        data,
        {"id": [1, 4], "name": ["John Doe", "Alice Smith"]},
        data,
    )


def test_approximate_deduplication_default_params2() -> None:
    assert_action_pipeline(
        ApproximateDeduplicator(),
        {
            "record_id": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "name": [
                "Acme Corp. Services",
                "Acme Services Corp.",
                "ACME Corp",
                "Smith, Alice M.",
                "Alice M. Smith",
                "Smith, Allan",
                "Bob's Garage",
                "Bobs Garage and Repair",
                "Bob's Garage",
            ],
            "address": [
                "123 Maple Street",
                "123 Maple St.",
                "125 Maple St.",
                "45 OAK",
                "45 Oak Street",
                "54 Oak St",
                "20 Industrial Ave",
                "20 Industrial Avenue",
                "20 Industrial Ave",
            ],
            "city": [
                "New York",
                "New York",
                "NYC",
                "LA",
                "Los Angeles",
                "LA",
                "Chicago",
                "Chicago",
                "Chicago",
            ],
        },
        {
            "record_id": [100, 101, 102, 103, 104, 105, 106],
            "name": [
                "Acme Corp. Services",
                "Acme Services Corp.",
                "ACME Corp",
                "Smith, Alice M.",
                "Alice M. Smith",
                "Smith, Allan",
                "Bob's Garage",
            ],
            "address": [
                "123 Maple Street",
                "123 Maple St.",
                "125 Maple St.",
                "45 OAK",
                "45 Oak Street",
                "54 Oak St",
                "20 Industrial Ave",
            ],
            "city": [
                "New York",
                "New York",
                "NYC",
                "LA",
                "Los Angeles",
                "LA",
                "Chicago",
            ],
        },
    )


def test_approximate_deduplication_custom_threshold() -> None:
    # Assuming threshold default is set low enough to catch these near matches.
    assert_action_pipeline(
        ApproximateDeduplicator(threshold=1.0),
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Alice Smith",
                "Bob J",
                "Bob Jones",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "London",
                "Paris",
                "Pariis",
            ],
            "value": [100, 100, 200, 200, 300, 300],
        },
        {
            "id": [1, 2, 3, 5, 6],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Bob J",
                "Bob Jones",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "Paris",
                "Pariis",
            ],
            "value": [100, 100, 200, 300, 300],
        },
    )


def test_ad_action_params_keep_last() -> None:
    assert_action_pipeline(
        ApproximateDeduplicator(keep="last"),
        {
            "id": [1, 2, 3, 4, 5, 6],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Alice Smith",
                "Bob J",
                "Bob Jones",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "London",
                "Paris",
                "Pariis",
            ],
            "value": [100, 100, 200, 200, 300, 300],
        },
        {
            "id": [1, 2, 4, 6],
            "name": [
                "John Doe",
                "Jhon Doe",
                "Alice Smith",
                "Bob Jones",
            ],
            "city": [
                "New York",
                "NY",
                "London",
                "Pariis",
            ],
            "value": [100, 100, 200, 300],
        },
    )
