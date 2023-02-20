import pandas as pd
import pytest

from tests.helpers import INPUT_DIR


@pytest.fixture
def iris_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR + "iris.csv")
