# Paul Escapil Sparse Grid for Resolution of Helmholtz equation in a cube

# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from bempp.api.space import Space

# other

import bempp.api
# from Helmholtz_Mean import PassageSurfaceMatrix, getFaceMatrix, NV3Dext, coord3DVext
import numpy as np
import operator
import time

class Neo_Space(Space):
    def __init__(self, grid, kind, order, L):
        self.L = L
        self.kind = kind
        self.order = order
        self.dof_dim = self.dof_dimension()
        super(Neo_Space, self).__init__(grid)
        if self.kind == 'P':
            self.h1D = 2**(-L)
        if self.kind == 'RT':
            self.h1D = 2**(1-L)
        if self.kind == 'P':
            self.N1D = 2**(L+1) - 1
        if self.kind == 'RT':
            self.N1D = 2**L
        self.elements = Elements(self)
        self.Unique = self.get_dimension()
        self.N = self.elements.N

    def dof_dimension(self):
        # Gives the type of dof (vertex, edge, point)
        if self.kind == 'P':
            return 2
        if self.kind == 'RT':
            return 1
        else:
            return 'Space_is_not_compatible'

    def get_dimension(self):
        u = self.grid.bounding_box
        h = np.zeros(3)
        for i in range(3):
            n = np.linalg.norm(u[0, i] - u[1, i])
            h[i] = n
        return h


class Elements():
    def __init__(self, Space):
        self.dof = list(Space.grid.leaf_view.entity_iterator(Space.dof_dim))
        self.dof_dimension = Space.dof_dim
        self.N = len(self.dof)
        self.Space = Space
        self.coord = self.get_coord()
        self.id = self.get_unique_id()

    def get_coord(self):
        # For Polynomial spaces, gives the coordinates of each vertex
        # For RT spaces gives the center of each edge
        N = self.N
        h = self.Space.h1D
        coord_list = self.dof
        coord = np.zeros([N, 3])
        if self.dof_dimension == 2:
            for i in range(N):
                coord[i, :] = coord_list[i].geometry.corners.T

        else:
            for i in range(N):
                corner0 = coord_list[i].geometry.corners[:, 0]
                corner1 = coord_list[i].geometry.corners[:, 1]
                center = 1/2 * (corner0 + corner1)
                coord[i, :] = center

        return np.around(coord/h)*h

    def get_unique_id(self, vert=None):
        # Just with 'P'
        if vert is None:
            vertices = self.coord
        else:
            vertices = vert
        h = self.Space.h1D
        N = self.Space.N1D + 2
        if len(vertices.shape) == 1:
            vertices = np.array([vertices])
        Number = vertices.shape[0]

        sol = np.zeros(Number)
        for i in range(Number):
            x = vertices[i, 0]
            y = vertices[i, 1]
            z = vertices[i, 2]
            xx = np.floor(0.5*(x + 1)/h*2)
            yy = np.floor(0.5*(y + 1)/h*2)
            zz = np.floor(0.5*(z + 1)/h*2)
            sol[i] = N**2*zz + N*yy + xx
        return sol.astype(int)

    def go_0_to_L(self, space0):

        iter0 = space0.elements.get_coord()
        iterL = self.Space.elements.get_coord()
        
        ele0 = self.Space.elements.get_unique_id(iter0)
        ele1 = self.Space.elements.get_unique_id()

        points0 = np.zeros([len(ele0), 3], dtype=int)
        points0[:, 0] = ele0
        points0[:, 1] = range(len(ele0))
        points0[:, 2] = 0

        points1 = np.zeros([len(ele1), 3], dtype=int)
        points1[:, 0] = ele1
        points1[:, 1] = range(len(ele1))
        points1[:, 2] = 1

        points = np.concatenate([points0, points1])
        points_sorted = sorted(points, key=operator.itemgetter(0))
        points_sorted = np.reshape(points_sorted, [len(points_sorted), 3])
        new_list = []

        coeff = np.zeros([len(points0), 2], dtype=int)
        k = 0
        for i in range(len(points_sorted)-1):
            if points_sorted[i+1, 0] == points_sorted[i, 0]:
                new_list.append(points_sorted[i])
                new_list.append(points_sorted[i+1])
                coeff[k, 0] = points_sorted[i, 1]
                coeff[k, 1] = points_sorted[i+1, 1]
                k += 1
        new_list = np.reshape(new_list, [len(new_list), 3])
        # new_list = np.array(new_list, dtype=int)

        coeff = sorted(coeff, key=operator.itemgetter(0))
        coeff = np.reshape(coeff, [len(coeff), 2])
        coeff_fin = np.array(coeff[:, 1])
        return coeff_fin

    def int_to_coord(self, integer_vec=None):
        # Just works with 'P'
        if integer_vec is None:
            integer_vector = range(len(self.coord))
        else:
            integer_vector = integer_vec

        if len(integer_vec.shape) == 1:
            integer_vec = np.array([integer_vec])
        Number = integer_vector.shape[0]
        h = self.Space.h1D
        N = self.Space.N1D + 2
        sol = np.zeros([Number, 3])
        for i in range(Number):
            integer = integer_vector[i]
            zz = np.floor(integer/N**2)
            integer = integer-zz*N**2
            yy = np.floor(integer/N)
            integer = integer-yy*N
            xx = integer

            x01 = xx*h/2
            y01 = yy*h/2
            z01 = zz*h/2

            x = 2 * x01 - 1
            y = 2 * y01 - 1
            z = 2 * z01 - 1
            sol[i, 0] = x
            sol[i, 1] = y
            sol[i, 2] = z
        return sol

    def get_list_of_triangles_vertices(self):
        """
        Given a grid, returns each dof of the vertices associated with a given triangle for all the triangles of the mesh
        """
        elements0 = list(self.Space.grid.leaf_view.entity_iterator(0))
        Ntri = len(elements0)
        triangles0 = np.zeros([Ntri, 3])

        for i in range(Ntri):
            el0 = elements0[i]
            triangles0[i, :] = self.Space.get_global_dofs(el0)
        return triangles0


def Neo_function_space(grid, kind, order, L, domains=None, closed=True, strictly_on_segment=False,
                       reference_point_on_segment=True, element_on_segment=False):

    from bempp.core.space.space import function_space as _function_space
    space = Neo_Space(_function_space(grid._impl, kind, order, domains, closed, strictly_on_segment,
                      reference_point_on_segment, element_on_segment), kind, order, L)

    return space


def get_unique_dof_id(self, vert=None):
    # Just with 'P'
    if vert is None:
        return np.array(range(len(self.coord)))
    else:
        vertices = vert
    if len(vertices.shape) == 1:
        vertices = np.array([vertices])
    Number = vertices.shape[0]
    sol = np.zeros(Number)
    known_vertices = self.coord
    vertices = vertices
    for i in range(Number):
        k = 0
        while k < len(known_vertices):
            if np.all(vertices[i] == known_vertices[k]):
                sol[i] = k
            if k == len(known_vertices):
                return 'error for %', vertices[i]
            k += 1
    return sol.astype(int)

"""
Lref = 7

gridL = bempp.api.import_grid('../gmsh/square_%s.msh' % (2))
spaceL = Neo_function_space(gridL, "P", 1, 1)

grid0 = bempp.api.import_grid('../gmsh/square_%s.msh' % (1))
space0 = Neo_function_space(grid0, "P", 1, 0)

iter0 = space0.elements.get_coord()
iterL = spaceL.elements.get_coord()

t1 = time.time()
print 'go oldy'
self = spaceL.elements
iter1 = get_unique_dof_id(self, iter0)

print time.time()-t1, 'oldy'
print '--------------'
t1 = time.time()


t1 = time.time()
print 'go new'
self = spaceL.elements
coeff_fin = spaceL.elements.go_0_to_L(space0)

print time.time()-t1, 'new'
t1 = time.time()
print '--------------'
print np.linalg.norm(coeff_fin-iter1), 'error'
"""
