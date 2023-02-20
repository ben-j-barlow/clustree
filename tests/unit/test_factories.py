import copy

from pairing_functions import szudzik

from clustree._factories import ClustreeConfig as cfg

test_setup_config = {
    "init": False,
    "sample_info": False,
    "image": False,
    "color": False,
}


def test_init_cf(iris_data):
    # TODO: find out if copy is needed
    setup_cf = copy.copy(test_setup_config)
    setup_cf["init"] = True
    cf = cfg(
        kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf
    )
    assert cf.node_cf[cfg.hash_k_k(1, 1)]["k_lower"] == 1
    assert cf.node_cf[cfg.hash_k_k(1, 1)]["k_upper"] == 1

    assert cf.node_cf[cfg.hash_k_k(2, 1)]["k_lower"] == 1
    assert cf.node_cf[cfg.hash_k_k(2, 2)]["k_lower"] == 2
    assert cf.node_cf[cfg.hash_k_k(2, 1)]["k_upper"] == 2
    assert cf.node_cf[cfg.hash_k_k(2, 2)]["k_upper"] == 2

    assert cf.node_cf[cfg.hash_k_k(3, 1)]["k_lower"] == 1
    assert cf.node_cf[cfg.hash_k_k(3, 2)]["k_lower"] == 2
    assert cf.node_cf[cfg.hash_k_k(3, 3)]["k_lower"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 1)]["k_upper"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 2)]["k_upper"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 3)]["k_upper"] == 3


def test_set_sample_information_node(iris_data):
    setup_cf = test_setup_config
    setup_cf["sample_info"] = True
    cf = cfg(
        kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf
    )
    assert len(cf.node_cf) == 6

    assert cf.node_cf[cfg.hash_k_k(1, 1)]["samples"] == 150

    assert cf.node_cf[cfg.hash_k_k(2, 1)]["samples"] == 70
    assert cf.node_cf[cfg.hash_k_k(2, 2)]["samples"] == 80

    assert cf.node_cf[cfg.hash_k_k(3, 1)]["samples"] == 45
    assert cf.node_cf[cfg.hash_k_k(3, 2)]["samples"] == 45
    assert cf.node_cf[cfg.hash_k_k(3, 3)]["samples"] == 60


def test_set_sample_information_edge(iris_data):
    setup_cf = test_setup_config
    setup_cf["sample_info"] = True
    cf = cfg(
        kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf
    )
    assert len(cf.edge_cf) == 6

    # samples and alpha
    assert cf.edge_cf[cfg.hash_k_k(2, 1, 1)]["samples"] == 70
    assert cf.edge_cf[cfg.hash_k_k(2, 2, 1)]["samples"] == 80

    assert cf.edge_cf[cfg.hash_k_k(2, 1, 1)]["alpha"] == 1
    assert cf.edge_cf[cfg.hash_k_k(2, 2, 1)]["alpha"] == 1

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["samples"] == 45
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["samples"] == 25
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["samples"] == 20
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["samples"] == 60

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["alpha"] == 1
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["alpha"] == 5 / 9
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["alpha"] == 4 / 9
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["alpha"] == 1

    # start and end
    assert cf.edge_cf[cfg.hash_k_k(2, 2, 1)]["start"] == cfg.hash_k_k(1, 1)
    assert cf.edge_cf[cfg.hash_k_k(2, 1, 1)]["end"] == cfg.hash_k_k(2, 1)

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["end"] == cfg.hash_k_k(3, 1)
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["end"] == cfg.hash_k_k(3, 2)
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["end"] == cfg.hash_k_k(3, 2)
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["end"] == cfg.hash_k_k(3, 3)

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["start"] == cfg.hash_k_k(2, 1)
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["start"] == cfg.hash_k_k(2, 1)
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["start"] == cfg.hash_k_k(2, 2)
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["start"] == cfg.hash_k_k(2, 2)


    assert cf.edge_cf[cfg.hash_k_k(2, 1, 1)]["alpha"] == 1
    assert cf.edge_cf[cfg.hash_k_k(2, 2, 1)]["alpha"] == 1

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["samples"] == 45
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["samples"] == 25
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["samples"] == 20
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["samples"] == 60

    assert cf.edge_cf[cfg.hash_k_k(3, 1, 1)]["alpha"] == 1
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 1)]["alpha"] == 5 / 9
    assert cf.edge_cf[cfg.hash_k_k(3, 2, 2)]["alpha"] == 4 / 9
    assert cf.edge_cf[cfg.hash_k_k(3, 3, 2)]["alpha"] == 1


def test_hash_k_k():
    assert szudzik.pair(1, 2, 3) == cfg.hash_k_k(1, 2, 3)
    assert szudzik.pair(1, 2) == cfg.hash_k_k(1, 2)
