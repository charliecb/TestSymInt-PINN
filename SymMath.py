from sympy import *

def gradient(exp, mat):
    diffs = []
    for i in range(mat.shape[0]):
        diffs.append([])
        for j in range(mat.shape[1]):
            diffs[-1].append(diff(exp, mat[i,j]))
    return Matrix(diffs)

def efficiently_integrate(integrand, var):
    coefficient_dict = integrand.expand().as_coefficients_dict()
    integrated_terms = []
    for term in coefficient_dict:
        integrated_terms.append(
            coefficient_dict[term]*integrate(term, var)
        )
    return sum(integrated_terms)
