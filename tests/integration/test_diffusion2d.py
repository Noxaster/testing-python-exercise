"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import pytest
import numpy as np

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()

    w = 20.
    h = 10.
    dx = 0.5
    dy = 0.2
    d = 5.

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=d, T_cold=200., T_hot=800.)

    expected_dt = (dx ** 2 * dy ** 2) / (2 * d * (dx ** 2 + dy ** 2))
    assert solver.dt == pytest.approx(expected_dt)

def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()

    w = 20.
    h = 10.
    dx = 0.5
    dy = 0.2
    d = 5.
    T_cold = 200.
    T_hot = 800.
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    nx = int(w / dx)
    ny = int(h / dy)

    solver.initialize_domain(w=w, h=h, dx=dx, dy=dy)
    solver.initialize_physical_parameters(d=d, T_cold=T_cold, T_hot=T_hot)

    u_actual = solver.set_initial_condition()
    u_expected = T_cold * np.ones((nx, ny))

    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u_expected[i, j] = T_hot

    assert np.allclose(u_actual, u_expected), "Integration test failed with set_initial_condition."