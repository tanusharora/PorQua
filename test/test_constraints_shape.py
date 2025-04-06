import os
import sys
import pytest
import numpy as np

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import load_data_msci
from constraints import Constraints


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

    # Convert lower/upper bounds to inequality constraints
    GhAb = constraints.to_GhAb(lbub_to_G=True)

    # --- Inequality constraints ---
    assert GhAb['G'] is not None
    assert GhAb['h'] is not None
    assert GhAb['G'].shape[1] == len(universe)
    assert GhAb['h'].shape[0] == GhAb['G'].shape[0]

    # --- Equality constraints ---
    assert GhAb['A'] is not None
    assert GhAb['b'] is not None
    assert GhAb['A'].shape[1] == len(universe)

    # Use np.atleast_1d to safely reshape scalar to array if needed
    b = np.atleast_1d(GhAb['b'])
    assert b.shape[0] == GhAb['A'].shape[0]
