# -*- coding: utf-8 -*-from __future__ import division
from Parameters import *
import time
from Scattering import Helmholtz_Scattering, Helmholtz_Variance
from Geometry import Sphere, Square
from Interpolator import Triangular_Interpolation, Passage_Matrix
from Helmholtz_Scattering_Sphere import Error_GridFunction, Helmholtz_Exact_Solution
from Tensor_Operator import vectensvec, DenseTensorLinearOperator
from New_Space import Neo_function_space
from Kernel import Custom_Kernel
from Matern import Matern


def main_final(L0, L):
    K0 = k_define()
    #Error = Helmholtz_Exact_Solution(K0)
    test = Matern(1 ,0.5 , 100, k = K0)
    Kernel = test.Matern_Kernel

    print 'go'
    print 'Assemblage de AL'


    t0 = time.time()
    Nl = []

    sp = []
    sp_square = []
    Ntotal = 0
    A = []
    P = []


    for l in range(0, L+1):
        grid = Square(l+1)
        sp_square.append(Neo_function_space(grid, 'P', 1, l))
        grid = Sphere(l+1)
        sp.append(Neo_function_space(grid, 'P', 1, l))
        sp_0 = sp[l]
        V = bem.operators.boundary.helmholtz.single_layer(sp_0, sp_0, sp_0, K0)
        A.append(V.weak_form())
        Nl.append(A[l].shape[0])

    print 'AL assemblé'
    print 'Assemblage des Al',

    for l in range(0, L+1):
        Ptemp = Passage_Matrix(sp_square[l], sp_square[L])
        P.append(Ptemp)



    N = Nl[L]

    uu_final = np.zeros([N, N], dtype=np.complex128)
    for l in range(0, L+1-L0):
        l0 = L0+l
        l1 = L-l
        if l0 <= l1:

            n0 = Nl[L0+l]
            n1 = Nl[L-l]
            
            sp0 = sp[l0]
            sp1 = sp[l1]

            C = Custom_Kernel(sp0, sp1, Kernel).reshape([n0 * n1])

            uu = Helmholtz_Variance(A[l0], A[l1], C)
            P0 = P[l0]
            P1 = P[l1]

            uu_out = np.dot(np.dot(P0.T, uu), P1)
            uu_final += uu_out
            Ntotal += n0 * n1

            if l0 != l1:
                uu_final += uu_out.T


    for l in range(L-L0):
        l0 = L0+l
        l1 = L-l-1
        if l0 <= l1:

            n0 = Nl[L0+l]
            n1 = Nl[L-l-1]
            
            sp0 = sp[l0]
            sp1 = sp[l1]

            C = Custom_Kernel(sp0, sp1, Kernel).reshape([n0 * n1])

            uu = Helmholtz_Variance(A[l0], A[l1], C)
            P0 = P[l0]
            P1 = P[l1]

            uu_out = np.dot(np.dot(P0.T, uu), P1)
            uu_final -= uu_out
            Ntotal += n0 * n1

            if l0 != l1:
                uu_final -= uu_out.T


    C = Custom_Kernel(sp[L], sp[L], Kernel)

    uu_full = Helmholtz_Variance(A[L], A[L], C.reshape(N*N))

    print 'Résultats otenus'
    func = test.Tensor

    points = grid.leaf_view.vertices.T
    Csol = np.zeros([N, N], dtype=np.complex128)
    for i in range(N):
        print i
        pi = points[i, :]
        for j in range(N):
            pj = points[j, :]
            Csol[i, j] = func(pi, pj)
    uu_exact = Csol

    spaceL = sp[L]

    variance_full = uu_full.diagonal()
    variance_final = uu_final.diagonal()
    variance_exact = uu_exact.diagonal()
    variance_full_gmsh = gf(spaceL, coefficients=variance_full)
    variance_final_gmsh = gf(spaceL, coefficients=variance_final)
    variance_exact_gmsh = gf(spaceL, coefficients=variance_exact)

    variance_error_full = gf(spaceL, coefficients=variance_exact-variance_full)
    variance_error_final = gf(spaceL, coefficients=variance_exact-variance_final)
    res = []
    res.append(variance_error_full.l2_norm()/variance_exact_gmsh.l2_norm())
    res.append(variance_error_final.l2_norm()/variance_exact_gmsh.l2_norm())

    # Hop = bempp.api.operators.boundary.laplace.hypersingular(spaceL, spaceL, spaceL)

    # res.append(np.abs(h_a_half(variance_error_full, Hop)/h_a_half(variance_exact_gmsh, Hop)))
    # res.append(np.abs(h_a_half(variance_error_final, Hop)/h_a_half(variance_exact_gmsh, Hop)))

    res.append(0)
    res.append(0)

    res.append(norm(uu_full.diagonal()-uu_exact.diagonal())/norm(uu_exact.diagonal()))
    res.append(norm(uu_final.diagonal()-uu_exact.diagonal())/norm(uu_exact.diagonal()))

    res.append(norm(uu_full-uu_exact)/norm(uu_exact))
    res.append(norm(uu_final-uu_exact)/norm(uu_exact))
    res.append(norm(uu_full-uu_final)/norm(uu_full))

    # normex = L2xL2_norm(uu_exact, spaceL)

    # res.append(L2xL2_norm(uu_full-uu_exact, spaceL)/normex)
    # res.append(L2xL2_norm(uu_final-uu_exact, spaceL)/normex)
    res.append(0)
    res.append(0)
    # res.append(0)
    res.append(Ntotal)
    res.append(uu_exact.shape[0]*uu_exact.shape[1])


    print ''
    print ''
    print ''
    print res[0], 'relative error_full L2'
    print res[1], 'relative error_sparse_final L2'
    print '--------------------------------------'

    print res[2], 'error_full H_0.5'
    print res[3], 'error_sparse_final H_0.5'
    print '--------------------------------------'

    print res[4], 'error_full l2 variance'

    print res[5], 'error_final l2 variance'

    print '--------------------------------------'
    print res[6], 'error_full l2'

    print res[7], 'error_final l2'

    print res[8], 'error_final-full l2'

    print '--------------------------------------'

    print res[9], 'error_full L2xL2'

    print res[10], 'error_final L2xL2'

    print '--------------------------------------'

    print res[11], 'DOF Sparse'
    print res[12], 'DOF Full'

    return res

