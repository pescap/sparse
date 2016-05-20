# -*- coding: utf-8 -*-from __future__ import division

from Parameters import *
import time
from Kernel import Custom_Kernel
from Geometry import Sphere, Square
from scipy.sparse.linalg import gmres
from Helmholtz_Scattering_Sphere import Error_GridFunction, Helmholtz_Exact_Solution
from Interpolator import Passage_Matrix
from New_Space import Neo_function_space

K0 = 10#k_define()

bem.global_parameters.assembly.boundary_operator_assembly_type = 'dense'

def combined_data_EFIE(x, n, domain_index, result):
    result[0] = np.exp(1j * K0 * x[0])

L0 = 0
gr0 = Sphere(L0+1)
gr0 = Square(L0+1)

sp0 = fs(gr0)

grid_fun0 = gf(sp0, fun=combined_data_EFIE)
b0 = grid_fun0.projections()


single_layer0 = bem.operators.boundary.helmholtz.single_layer(sp0, sp0, sp0, K0)
V0 = single_layer0.weak_form()

neumann_fun0, info_conv = gmres(V0, b0, tol=tolerance())#, callback=make_callback())

u0 = gf(sp0, coefficients=neumann_fun0)

# MAINTENANT SOL SUR AUTRE


L = 1
print 'xxxx'
print L0, L, '=L0, L'
print 'xxxx'
gr1 = Sphere(L+1)
gr1 = Square(L+1)
sp1 = fs(gr1)

grid_fun1 = gf(sp1, fun=combined_data_EFIE)
b1 = grid_fun1.projections()


single_layer1 = bem.operators.boundary.helmholtz.single_layer(sp1, sp1, sp1, K0)
V1 = single_layer1.weak_form()

neumann_fun1, info_conv = gmres(V1, b1, tol=tolerance())#, callback=make_callback())

u1 = gf(sp1, coefficients=neumann_fun1)

### Maintenant le faire passer sur le niveau u0

P = []
sp_square = []
for l in range(L+1):
    gr = Square(l+1)
    sp_square.append(Neo_function_space(gr, 'P', 1, l))

for l in range(L+1):
    Ptemp = Passage_Matrix(sp_square[l], sp_square[L])
    P.append(Ptemp)


Pas = P[L0]

Pm = np.linalg.pinv(Pas)

Atemp = V1.matmat(Pas.T)
from scipy.sparse.linalg import aslinearoperator
An0 = np.dot(Pas, Atemp)
An0 = aslinearoperator(An0)

bn0 = np.dot(Pas, b1)

grS0 = Square(L0+1)
grS1 = Square(L+1)
spS0 = Neo_function_space(grS0, 'P', 1, L0)
spS1 = Neo_function_space(grS1, 'P', 1, L)

iter0 = spS0.elements.get_coord()
iter1 = spS1.elements.go_0_to_L(spS0) 
sol = np.zeros(b0.shape)
sol = u1.coefficients[iter1]
u0neo = gf(sp0, coefficients=sol)


"""
neumann_funn0, info_conv = gmres(An0, bn0, tol=tolerance(), callback=make_callback())
un0 = gf(sp0, coefficients=neumann_funn0)

bn0m = np.dot(b1, Pm)
Atempm = V1.matmat(Pm)
An0m = np.dot(Pm.T, Atempm)
An0m = aslinearoperator(An0m)

neumann_funn0m, info_conv = gmres(An0m, bn0m, tol=tolerance(), callback=make_callback())
un0m = gf(sp0, coefficients=neumann_funn0m)

# ERRORS:

Error = Helmholtz_Exact_Solution(K0)

solex = Error.uBoundaryNeumannTrace

print '---------'
err0 = u0.relative_error(solex)
print err0, 'error 0'

errn0 = un0.relative_error(solex)
print errn0, 'error n0, TRANSPOSE'


errn0m = un0m.relative_error(solex)
print errn0m, 'error n0m, INVERSE'


err1 = u1.relative_error(solex)
print err1, 'error 1'
"""

Error = Helmholtz_Exact_Solution(K0)

solex = Error.uBoundaryNeumannTrace

print '---------'

c1 = u1.coefficients

u0t = gf(sp0, coefficients=np.dot(Pas, c1))
u0m = gf(sp0, coefficients=np.dot(c1, Pm))


err0= u0.relative_error(solex)
print err0, 'error 0'
err1 = u1.relative_error(solex)
print err1, 'error 1'
print '---------'

err0t = u0t.relative_error(solex)
print err0t, 'error 0 T'
err0m = u0m.relative_error(solex)
print err0m, 'error 0 M '
err0neo = u0neo.relative_error(solex)
print err0neo, 'error 0 neo '



