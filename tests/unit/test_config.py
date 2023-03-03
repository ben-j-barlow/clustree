import copy

import matplotlib as mpl
import pytest

from clustree._config import ClustreeConfig as cfg
from clustree._config import CONTROL_LIST, data_to_color
from clustree._hash import hash_edge_id, hash_node_id

DEFAULT_CONFIG = {k: False for k in CONTROL_LIST}


def test_init_cf(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf["init"] = True
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
    assert cf.node_cf[hash_node_id(1, 1)]["k"] == 1
    assert cf.node_cf[hash_node_id(1, 1)]["res"] == 1

    assert cf.node_cf[hash_node_id(2, 1)]["k"] == 1
    assert cf.node_cf[hash_node_id(2, 2)]["k"] == 2
    assert cf.node_cf[hash_node_id(2, 1)]["res"] == 2
    assert cf.node_cf[hash_node_id(2, 2)]["res"] == 2

    assert cf.node_cf[hash_node_id(3, 1)]["k"] == 1
    assert cf.node_cf[hash_node_id(3, 2)]["k"] == 2
    assert cf.node_cf[hash_node_id(3, 3)]["k"] == 3
    assert cf.node_cf[hash_node_id(3, 1)]["res"] == 3
    assert cf.node_cf[hash_node_id(3, 2)]["res"] == 3
    assert cf.node_cf[hash_node_id(3, 3)]["res"] == 3


def test_set_sample_information_node(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf["sample_info"] = True
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
    assert len(cf.node_cf) == 6

    assert cf.node_cf[hash_node_id(1, 1)]["samples"] == 150

    assert cf.node_cf[hash_node_id(2, 1)]["samples"] == 70
    assert cf.node_cf[hash_node_id(2, 2)]["samples"] == 80

    assert cf.node_cf[hash_node_id(3, 1)]["samples"] == 45
    assert cf.node_cf[hash_node_id(3, 2)]["samples"] == 45
    assert cf.node_cf[hash_node_id(3, 3)]["samples"] == 60


def test_set_sample_information_edge(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf["sample_info"] = True
    cf = cfg(kk=3, prefix="K", data=iris_data, image_cf=None, _setup_cf=setup_cf)
    assert len(cf.edge_cf) == 6

    # samples and alpha
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=1, k_start=1)]["samples"] == 70
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=2, k_start=1)]["samples"] == 80

    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=1, k_start=1)]["alpha"] == 1
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=2, k_start=1)]["alpha"] == 1

    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)]["samples"] == 45
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)]["samples"] == 25
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)]["samples"] == 20
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)]["samples"] == 60

    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)]["alpha"] == 1
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)]["alpha"] == 5 / 9
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)]["alpha"] == 4 / 9
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)]["alpha"] == 1

    # start and end
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=2, k_start=1)][
        "start"
    ] == hash_node_id(1, 1)
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=1, k_start=1)][
        "end"
    ] == hash_node_id(2, 1)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)][
        "end"
    ] == hash_node_id(3, 1)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)][
        "end"
    ] == hash_node_id(3, 2)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)][
        "end"
    ] == hash_node_id(3, 2)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)][
        "end"
    ] == hash_node_id(3, 3)

    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)][
        "start"
    ] == hash_node_id(2, 1)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)][
        "start"
    ] == hash_node_id(2, 1)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)][
        "start"
    ] == hash_node_id(2, 2)
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)][
        "start"
    ] == hash_node_id(2, 2)

    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=1, k_start=1)]["alpha"] == 1
    assert cf.edge_cf[hash_edge_id(k_upper=2, k_end=2, k_start=1)]["alpha"] == 1

    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)]["samples"] == 45
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)]["samples"] == 25
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)]["samples"] == 20
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)]["samples"] == 60

    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=1, k_start=1)]["alpha"] == 1
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=1)]["alpha"] == 5 / 9
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=2, k_start=2)]["alpha"] == 4 / 9
    assert cf.edge_cf[hash_edge_id(k_upper=3, k_end=3, k_start=2)]["alpha"] == 1


def test_set_node_color_prefix(iris_data):
    setup_cf = DEFAULT_CONFIG
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
    setup_cf = DEFAULT_CONFIG
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
        hash_node_id(k_upper=k_upper, k_lower=k_lower)
        for k_upper in range(1, kk + 1)
        for k_lower in range(1, k_upper + 1)
    ]
    samples = [150, 70, 80, 45, 45, 60]
    exp_color = data_to_color(
        data={k: v for k, v in zip(node_id, samples)},
        cmap=mpl.cm.Blues,
        return_sm=False,
    )

    # actual
    act_color = {k: v["node_color"] for k, v in cf.node_cf.items()}
    assert all([isinstance(v["node_color"], tuple) for k, v in cf.node_cf.items()])
    assert exp_color == act_color


def test_set_node_color_agg(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf.update({"sample_info": True, "node_color": True})

    # produce expected
    kk = 3
    node_id = [
        hash_node_id(k_upper=k_upper, k_lower=k_lower)
        for k_upper in range(1, kk + 1)
        for k_lower in range(1, k_upper + 1)
    ]
    agg_res = [876.5, 369.8, 506.7, 225.5, 265.2, 385.8]
    exp_color = data_to_color(
        data={k: v for k, v in zip(node_id, agg_res)},
        cmap=mpl.cm.Blues,
        return_sm=False,
    )

    # actual: with agg as callable
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
    act_color = {k: v["node_color"] for k, v in cf.node_cf.items()}
    assert all([isinstance(v["node_color"], tuple) for k, v in cf.node_cf.items()])
    assert exp_color == act_color

    # actual: with agg as str
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        node_color="sepal_length",
        node_color_aggr="sum",
        node_cmap=mpl.cm.Blues,
    )
    act_color = {k: v["node_color"] for k, v in cf.node_cf.items()}
    assert all([isinstance(v["node_color"], tuple) for k, v in cf.node_cf.items()])
    assert exp_color == act_color


def test_set_node_color_no_agg_chosen(iris_data):
    setup_cf = DEFAULT_CONFIG
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
    setup_cf = DEFAULT_CONFIG
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


def test_set_edge_color_prefix(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf.update({"sample_info": True, "edge_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        edge_color="K",
    )
    exp = [f"C{res}" for res in [2, 2, 3, 3, 3, 3]]
    assert [cf.edge_cf[k]["edge_color"] for k in cf.edge_cf] == exp


def test_set_edge_color_samples(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf.update({"sample_info": True, "edge_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        edge_color="samples",
    )
    act_color = [v["edge_color"] for k, v in cf.edge_cf.items()]
    edge_id = [
        hash_edge_id(k_upper=2, k_end=1, k_start=1),
        hash_edge_id(k_upper=2, k_end=2, k_start=1),
        hash_edge_id(k_upper=3, k_end=1, k_start=1),
        hash_edge_id(k_upper=3, k_end=2, k_start=1),
        hash_edge_id(k_upper=3, k_end=2, k_start=2),
        hash_edge_id(k_upper=3, k_end=3, k_start=2),
    ]

    samples = [70, 80, 45, 25, 20, 60]

    exp_color = data_to_color(
        data={k: v for k, v in zip(edge_id, samples)},
        cmap=mpl.cm.Reds,
        return_sm=False,
    )

    # actual
    act_color = {k: v["edge_color"] for k, v in cf.edge_cf.items()}
    assert all([isinstance(v["edge_color"], tuple) for k, v in cf.edge_cf.items()])
    assert exp_color == act_color


def test_set_edge_color_fixed(iris_data):
    setup_cf = DEFAULT_CONFIG
    setup_cf.update({"sample_info": True, "edge_color": True})
    cf = cfg(
        kk=3,
        prefix="K",
        data=iris_data,
        image_cf=None,
        _setup_cf=setup_cf,
        edge_color="C1",
    )
    act_color = [v["edge_color"] for k, v in cf.edge_cf.items()]
    assert all([isinstance(v["edge_color"], str) for k, v in cf.edge_cf.items()])
    assert act_color == ["C1" for _ in range(6)]
