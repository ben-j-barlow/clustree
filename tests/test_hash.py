from pairing_functions import szudzik

from clustree._hash import hash_edge_id, hash_node_id


def test_hash_node_id():
    assert hash_node_id(k_lower=1, k_upper=4) == szudzik.pair(4, 1)


def test_hash_edge_id():
    assert hash_edge_id(k_upper=5, k_start=3, k_end=4) == szudzik.pair(5, 3, 4)
