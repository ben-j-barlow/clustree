from pairing_functions import szudzik


def hash_node_id(k_upper: int, k_lower: int) -> int:
    return szudzik.pair(k_upper, k_lower)


def hash_edge_id(k_upper: int, k_start: int, k_end: int) -> int:
    return szudzik.pair(k_upper, k_start, k_end)
