# PART 8: Compare the numerical and analytical solutions #######################

# Define functions that evaluate the Dirichlet and Neumann trace of the
# analytical solution, expressed as a series of spherical wave functions

from __future__ import division
from Parameters import *
from scipy.special import sph_jn, sph_yn, lpn


class Helmholtz_Exact_Solution():
    """Interface for getting exact solution of a soft sphere
    scattering problem with Helhmoltz equations

    Parameters:
    k : wavenumber
    """

    def __init__(self, k):
        self.k = k
        kExt = k

        l_max = 200
        l = np.arange(0, l_max + 1)

        jExt, djExt = sph_jn(l_max, kExt)
        yExt, dyExt = sph_yn(l_max, kExt)
        hExt = jExt + 1j * yExt


        aInc = (2 * l + 1) * 1j ** l

        np.seterr(divide='ignore',  invalid='ignore')
        cBound = 1/(1j * kExt) / hExt
        cDir = jExt / hExt
        cBoundSq = (2l + 1) * 1j ** (l-2) / (hExt**2) / (kExt**2) / jExt 
        for l in range(l_max + 1):
            if abs(cBound[l]) < 1e-16:
                # neglect all further terms
                l_max = l - 1
                aInc = aInc[:l]
                cBound = cBound[:l]
                cBoundSq = cBoundSq[:l]
                cDir = cDir[:l]
                break

        self.cDir = cDir
        self.aInc = aInc
        self.cBound = cBound
        self.l_max = l_max
        self.cBoundSq = cBoundSq


    # Verificar los valores para el uExactDirichlet...

    def uExactDirichletTrace(self, point):
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        hD, dhD = sph_jn(self.l_max, self.kExt * r) + 1j*sph_yn(self.l_max, self.kExt * r)
        Y, dY = lpn(self.l_max, x / r)
        return (self.cDir * hD * Y).sum()

    def uExactBoundaryNeumannTrace(self, point, normal, domain_index, result):
        x, y, z = point
        n_x, n_y, n_z = normal
        r = np.sqrt(x**2 + y**2 + z**2)
        Y, dY = lpn(self.l_max, x / r)
        result[0] = (self.aInc * self.cBound * Y).sum()
        #return result

    def uBoundaryNeumannTrace(self, point):
        x, y, z = point
        r = np.sqrt(x**2 + y**2 + z**2)
        Y, dY = lpn(self.l_max, x / r)
        val = (self.aInc * self.cBound * Y).sum()
        return val

    def uExactSquaredBoundaryNeumannTrace(self, point, normal, domain_index, result):
        x, y, z = point
        n_x, n_y, n_z = normal
        r = np.sqrt(x**2 + y**2 + z**2)
        Y, dY = lpn(self.l_max, x / r)
        result[0] = (self.cBoundSq * Y).sum()
        #return result

class Error_GridFunction():
    def __init__(self, Test_Solution, k):
        space = Test_Solution.space
        self.Exact_Solution = Helmholtz_Exact_Solution(k)
        self.k = k
        self.space = space
        self.Test_Grid = Test_Solution
        sol = GridFunction(space, fun=self.Exact_Solution.uExactBoundaryNeumannTrace)
        self.Exact_Grid = sol
        err = sol - Test_Solution
        self.Error_Grid = err
        self.coefficients = self.Error_Grid.coefficients
        relative_coefficients = self.Error_Grid.coefficients / self.Exact_Grid.coefficients
        relative_coefficients = np.abs(np.real(relative_coefficients)) + 1j*np.abs(np.imag(relative_coefficients))
        self.Relative_Error_Grid = GridFunction(space, coefficients=relative_coefficients)

    def plot(self, relative=1):
        if relative == 0:
            toplot = self.Error_Grid
        else:
            toplot = self.Relative_Error_Grid
        toplot.plot()

    def l2_norm(self, relative=1):
        a = self.Error_Grid.l2_norm()
        if relative == 1:
            a = a/self.Exact_Grid.l2_norm()
        return a

    def h_m_a_half(self, relative=1):
        space = self.space
        Hop = bempp.api.operators.boundary.laplace.single_layer(space, space, space)
        err = self.Error_Grid
        err_real = bempp.api.GridFunction(space, coefficients=np.real(err.coefficients))
        err_imag = bempp.api.GridFunction(space, coefficients=np.imag(err.coefficients))
        h12 = np.sqrt(np.vdot(err_real.coefficients, (Hop * err_real).projections(space)) + 1j*np.vdot(err_real.coefficients, (Hop * err_imag).projections(space)))
        if relative == 1:
            exa = self.Exact_Grid
            exa_real = bempp.api.GridFunction(space, coefficients=np.real(exa.coefficients))
            exa_imag = bempp.api.GridFunction(space, coefficients=np.imag(exa.coefficients))
            div = np.sqrt(np.vdot(exa_real.coefficients, (Hop * exa_real).projections(space)) + 1j*np.vdot(exa_real.coefficients, (Hop * exa_imag).projections(space)))
            h12 = h12/div
        return h12

    def h_a_half(self, relative=1):
        space = self.space
        Hop = bempp.api.operators.boundary.laplace.hypersingular(space, space, space)
        err = self.Error_Grid
        err_real = bempp.api.GridFunction(space, coefficients=np.real(err.coefficients))
        err_imag = bempp.api.GridFunction(space, coefficients=np.imag(err.coefficients))
        h12 = np.sqrt(np.vdot(err_real.coefficients, (Hop * err_real).projections(space)) + 1j*np.vdot(err_real.coefficients, (Hop * err_imag).projections(space)))
        if relative == 1:
            exa = self.Exact_Grid
            exa_real = bempp.api.GridFunction(space, coefficients=np.real(exa.coefficients))
            exa_imag = bempp.api.GridFunction(space, coefficients=np.imag(exa.coefficients))
            div = np.sqrt(np.vdot(exa_real.coefficients, (Hop * exa_real).projections(space)) + 1j*np.vdot(exa_real.coefficients, (Hop * exa_imag).projections(space)))
            h12 = h12/div
        return h12

if __name__ == "__main__":
    from Scattering import Helmholtz_Scattering
    from Geometry import Sphere
    gr_0 = Sphere(2)
    sp_0 = fs(gr_0)

    Error = Helmholtz_Exact_Solution(k0)

    u_exact_gmsh = gf(sp_0, fun = Error.uExactBoundaryNeumannTrace)
    u_du = Helmholtz_Scattering(sp_0, k0, g=-u_exact_gmsh, combined=0, hmat=0, info=1)

    u_du.plot()


    u_exact_gmsh = gf(sp_0, fun = Error.uExactSquaredBoundaryNeumannTrace)

    u_exact_gmsh.plot()