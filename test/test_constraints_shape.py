import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure the src/ directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from constraints import Constraints
from data_loader import load_data_msci


@pytest.fixture
def sample_constraints():
    """
    Fixture to load MSCI data and initialize Constraints.
    """
    data = load_data_msci(os.path.join(os.getcwd(), 'data'))
    universe = data['return_series'].columns
    constraints = Constraints(selection=universe)
    return constraints, universe


def test_add_budget_and_box_constraints(sample_constraints):
    """
    Tests the structure of G, h, A, b matrices after applying budget and box constraints.
    """
    constraints, universe = sample_constraints
    constraints.add_budget()
    constraints.add_box("LongOnly")

    # Important: convert lower/upper bounds into G/h constraints
    GhAb = constraints.to_GhAb(lbub_to_G=True)

    # -- Inequality Constraints (G, h) --
    assert GhAb['G'] is not None, "G matrix is None"
    assert GhAb['h'] is not None, "h vector is None"
    assert GhAb['G'].shape[1] == len(universe), "Mismatch in G matrix column count"
    assert GhAb['h'].shape[0] == GhAb['G'].shape[0], "Mismatch between G and h shapes"

    # -- Equality Constraints (A, b) --
    assert GhAb['A'] is not None, "A matrix is None"
    assert GhAb['b'] is not None, "b vector is None"
    assert GhAb['A'].shape[1] == len(universe), "Mismatch in A matrix column count"
    assert GhAb['b'].shape[0] == GhAb['A'].shape[0], f"b shape {GhAb['b'].shape}, A shape {GhAb['A'].shape}"

