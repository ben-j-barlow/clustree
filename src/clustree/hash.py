from pairing_functions import szudzik


def hash_node_id(k_upper: int, k_lower: int) -> int:
    return szudzik.pair(k_upper, k_lower)


# def unhash_node_id(node_id: int) -> tuple[int, int]:
#     return szudzik.unpair(node_id)


def hash_edge_id(k_upper: int, k_start: int, k_end: int) -> int:
    return szudzik.pair(k_upper, k_start, k_end)


# def unhash_edge_id(edge_id: int) -> tuple[int, int, int]:
#     return szudzik.unpair(edge_id, n=3)
