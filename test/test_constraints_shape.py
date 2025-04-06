import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from constraints import Constraints
from data_loader import load_data_msci

@pytest.fixture
def sample_constraints():
    data = load_data_msci(os.path.join(os.getcwd(), 'data'))
    universe = data['return_series'].columns
    constraints = Constraints(selection=universe)
    return constraints, universe

def test_add_budget_and_box_constraints(sample_constraints):
    constraints, universe = sample_constraints
    constraints.add_budget()
    constraints.add_box("LongOnly")

    GhAb = constraints.to_GhAb()

    assert GhAb['A'].shape[1] == len(universe)
    assert GhAb['b'].shape[0] == GhAb['A'].shape[0]
    assert GhAb['G'].shape[1] == len(universe)
    assert GhAb['h'].shape[0] == GhAb['G'].shape[0]
