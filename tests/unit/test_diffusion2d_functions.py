"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    w = 20.
    h = 10.
    dx = 0.5
    dy = 0.2

    expected_nx = int(w / dx)
    expected_ny = int(h / dy)

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)

    assert solver.nx == expected_nx
    assert solver.ny == expected_ny

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()

    d = 5.
    T_cold = 200.
    T_hot = 800.
    dx = 0.2
    dy = 0.1

    solver.dx = dx
    solver.dy = dy
    dx2 = dx * dx
    dy2 = dy * dy
    expected_dt = dx2 * dy2 / (2 * d * (dx2 + dy2))

    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    assert solver.dt == pytest.approx(expected_dt)

def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.set_initial_condition
    """
    solver = SolveDiffusion2D()

    w = 20.
    h = 10.
    dx = 0.5
    dy = 0.2
    T_cold = 200.
    T_hot = 800.
    r, cx, cy = 2, 5, 5
    r2 = r ** 2

    solver.w = w
    solver.h = h
    solver.dx = dx
    solver.dy = dy
    solver.T_cold = T_cold
    solver.T_hot = T_hot
    solver.nx = int(w / dx)
    solver.ny = int(h / dy)

    u_actual = solver.set_initial_condition()
    u_expected = solver.T_cold * np.ones((solver.nx, solver.ny))

    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                u_expected[i, j] = solver.T_hot

    assert np.allclose(u_actual, u_expected), "Unit test failed with set_initial_condition."