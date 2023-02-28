import copy

import matplotlib as mpl
import pytest
from pairing_functions import szudzik

from clustree.config import ClustreeConfig as cfg
from clustree.config import _data_to_color, control_list

test_setup_config = {k: False for k in control_list}


def test_init_cf(iris_data):
    setup_cf = copy.copy(test_setup_config)
    setup_cf["init"] = True
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
    assert cf.node_cf[cfg.hash_k_k(1, 1)]["k"] == 1
    assert cf.node_cf[cfg.hash_k_k(1, 1)]["res"] == 1

    assert cf.node_cf[cfg.hash_k_k(2, 1)]["k"] == 1
    assert cf.node_cf[cfg.hash_k_k(2, 2)]["k"] == 2
    assert cf.node_cf[cfg.hash_k_k(2, 1)]["res"] == 2
    assert cf.node_cf[cfg.hash_k_k(2, 2)]["res"] == 2

    assert cf.node_cf[cfg.hash_k_k(3, 1)]["k"] == 1
    assert cf.node_cf[cfg.hash_k_k(3, 2)]["k"] == 2
    assert cf.node_cf[cfg.hash_k_k(3, 3)]["k"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 1)]["res"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 2)]["res"] == 3
    assert cf.node_cf[cfg.hash_k_k(3, 3)]["res"] == 3


def test_set_sample_information_node(iris_data):
    setup_cf = test_setup_config
    setup_cf["sample_info"] = True
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
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
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
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


def test_set_node_color_prefix(iris_data):
    setup_cf = test_setup_config
    setup_cf.update({"init": True, "node_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        node_color="K",
    )
    exp = [f"C{res}" for res in [1, 2, 2, 3, 3, 3]]
    assert [cf.node_cf[k]["node_color"] for k in cf.node_cf] == exp


def test_set_node_color_samples(iris_data):
    setup_cf = test_setup_config
    setup_cf.update({"sample_info": True, "node_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        node_color="samples",
        node_cmap=mpl.cm.Blues,
    )

    # produce expected
    kk = 3
    node_id = [
        cfg.hash_k_k(k_upper=k_upper, k_lower=k_lower)
        for k_upper in range(1, kk + 1)
        for k_lower in range(1, k_upper + 1)
    ]
    samples = [150, 70, 80, 45, 45, 60]
    exp_color = _data_to_color(
        data={k: v for k, v in zip(node_id, samples)},
        cmap=mpl.cm.Blues,
        return_sm=False,
    )

    # actual
    act_color = {k: v["node_color"] for k, v in cf.node_cf.items()}
    assert all([isinstance(v["node_color"], tuple) for k, v in cf.node_cf.items()])
    assert exp_color == act_color


def test_set_node_color_agg(iris_data):
    setup_cf = test_setup_config
    setup_cf.update({"sample_info": True, "node_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        node_color="sepal_length",
        node_color_aggr=sum,
        node_cmap=mpl.cm.Blues,
    )

    # produce expected
    kk = 3
    node_id = [
        cfg.hash_k_k(k_upper=k_upper, k_lower=k_lower)
        for k_upper in range(1, kk + 1)
        for k_lower in range(1, k_upper + 1)
    ]
    agg_res = [876.5, 369.8, 506.7, 225.5, 265.2, 385.8]
    exp_color = _data_to_color(
        data={k: v for k, v in zip(node_id, agg_res)},
        cmap=mpl.cm.Blues,
        return_sm=False,
    )

    # actual
    act_color = {k: v["node_color"] for k, v in cf.node_cf.items()}
    assert all([isinstance(v["node_color"], tuple) for k, v in cf.node_cf.items()])
    assert exp_color == act_color


def test_set_node_color_no_agg_chosen(iris_data):
    setup_cf = test_setup_config
    setup_cf.update({"sample_info": True, "node_color": True})
    with pytest.raises(ValueError):
        cf = cfg(
            kk=3,
            prefix="K",
            data=iris_data,
            image_cf=None,
            _setup_cf=setup_cf,
            node_color="sepal_length",
            node_cmap=mpl.cm.Blues,
        )
    with pytest.raises(ValueError):
        cf = cfg(
            kk=3,
            prefix="K",
            data=iris_data,
            image_cf=None,
            _setup_cf=setup_cf,
            node_color="sepal_length",
            node_color_aggr=None,
            node_cmap=mpl.cm.Blues,
        )


def test_set_node_color_fixed(iris_data):
    setup_cf = test_setup_config
    setup_cf.update({"sample_info": True, "node_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        node_color="C1",
    )
    act_color = [v["node_color"] for k, v in cf.node_cf.items()]
    assert all([isinstance(v["node_color"], str) for k, v in cf.node_cf.items()])
    assert act_color == ["C1" for _ in range(6)]


def test_hash_k_k():
    assert szudzik.pair(1, 2, 3) == cfg.hash_k_k(1, 2, 3)
    assert szudzik.pair(1, 2) == cfg.hash_k_k(1, 2)
