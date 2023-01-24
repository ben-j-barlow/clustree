def _get_img_name_pattern(kk: int) -> list[str]:
    """
    Given kk, produce list of form 'K_k'. \
    For example, kk = 2 produces ['1_1', '2_1', '2_2'].

    :param kk: int, highest cluster resolution, 1 or greater.
    :return: list, containing K_k combinations
    """
    return [f"{K}_{k}" for K in range(1, kk + 1) for k in range(1, K + 1)]
