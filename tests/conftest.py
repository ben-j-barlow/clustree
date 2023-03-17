import pandas as pd
import pytest

from tests.helpers import INPUT_DIR


@pytest.fixture
def iris_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR + "iris.csv")


@pytest.fixture
def iris_data_0() -> pd.DataFrame:
    data = pd.read_csv(INPUT_DIR + "iris.csv")
    for col in ["K1", "K2", "K3"]:
        data[col] -= 1
    return data
