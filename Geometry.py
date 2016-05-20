# Paul Escapil Sparse Grid for Resolution of Helmholtz equation in a cube

# -*- coding: utf-8 -*-

from __future__ import division
from Parameters import *
from bempp.api.grid import grid_from_element_data


def Square(L):
    gr0 = bem.import_grid('../gmsh/square_%s.msh' % 0)
    gr1 = gr0.clone()
    for i in range(L):
        elements0 = list(gr1.leaf_view.entity_iterator(0))
        for el0 in elements0:
            gr1.mark(el0)
        gr1.refine()
    return gr1


def Sphere(L):
    if type(L) == int:
        gr0 = Square(L)
        x, y, z = gr0.leaf_view.vertices.T[:, 0], gr0.leaf_view.vertices.T[:, 1], gr0.leaf_view.vertices.T[:, 2]
    else:
        x, y, z = L.T
    N = len(x)
    dx = x * np.sqrt(1.0 - (y*y/2.0) - (z*z/2.0) + (y*y*z*z/3.0))
    dy = y * np.sqrt(1.0 - (z*z/2.0) - (x*x/2.0) + (z*z*x*x/3.0))
    dz = z * np.sqrt(1.0 - (x*x/2.0) - (y*y/2.0) + (x*x*y*y/3.0))

    vertices = np.zeros([N, 3])
    vertices[:, 0] = dx
    vertices[:, 1] = dy
    vertices[:, 2] = dz
    vertices = vertices.T
    if type(L) == int:
        elements = gr0.leaf_view.elements
        grid = grid_from_element_data(vertices, elements)
    else:
        grid = vertices.T
    return grid

"""
def Sphere(L):
    gr0 = Square(L)
    vertices_0 = gr0.leaf_view.vertices.T
    elements = gr0.leaf_view.elements
    N1 = len(vertices_0)
    vertices_eps = np.zeros([N1, 3])
    for i in range(N1):
        d = norm(vertices_0[i, :])
        vertices_eps[i, :] = vertices_0[i, :]/d
    grid_eps = grid_from_element_data(vertices_eps.T, elements)
    
    return grid_eps
"""


def Perturbed_Space(sp0, Field, eps):
    gr0 = sp0.grid
    vertices_0 = gr0.leaf_view.vertices.T
    elements = gr0.leaf_view.elements
    N1 = len(vertices_0)
    Field = gf(sp0, fun=Field)
    lent = []
    vertices_eps = np.zeros([N1, 3])
    for i in range(N1):
        vertices_eps[i, :] = vertices_0[i, :] + eps * Field.coefficients[i]* vertices_0[i, :]/norm(vertices_0[i, :])
        lent.append(norm(vertices_eps[i, :])-1)

    grid_eps = grid_from_element_data(vertices_eps.T, elements)
    space_eps = fs(grid_eps)
    return space_eps


if __name__ == "__main__":
    # Test for Perturbed_Space
    from Geometry import Sphere
    L = 1
    a_0, a_1, a_2, a_3 = np.random.random(4)-1
    grid_0 = Sphere(L+1)
    space_0 = fs(grid_0)
    from numpy import sin
    from numpy.linalg import norm

    def U_field(point, n, domain_index, result):

        x, y, z = point
        d = x**2 + y**2 + z**2
        #a_0, a_1, a_2, a_3 = 1, 1, 1, 1
        result[0] = 1 / 7 * sin(7 * y) * a_0 + 1 / 5 * sin(5 * x) * a_1 * 1 / 9 * sin (9 * z) * a_2

    space_eps = Perturbed_Space(space_0, U_field, 0.1)

    space_eps.grid.plot()






