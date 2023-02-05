from clustree._handle_pars import from_k_k, to_k_k


def test_to_k_k():
    assert to_k_k(3, 5) == "3_5"
    assert to_k_k(3, [1, 2, 3]) == ["3_1", "3_2", "3_3"]
    assert to_k_k(0, 0) == "0_0"
    assert to_k_k(10000, 1000) == "10000_1000"
    assert to_k_k(10000, [1000, 2000, 3000]) == [
        "10000_1000",
        "10000_2000",
        "10000_3000",
    ]


def test_from_k_k():
    assert from_k_k("K_5") == 5
    assert from_k_k(["K_1", "K_2", "K_3"]) == [1, 2, 3]
    assert from_k_k("K_0") == 0
    assert from_k_k(["K_0", "K_0", "K_0"]) == [0, 0, 0]
    assert from_k_k(["K_5", "K_7", "K_9"]) == [5, 7, 9]
