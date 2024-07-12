import numpy as np
import sympy as sp
import scalingFunctionsHelperPy as doh

# Define symbols
rho = 5
N = 4
h1 = sp.Symbol('h1')
h2 = sp.Symbol('h2')
h3 = sp.Symbol('h3')
h4 = sp.Symbol('h4')

h_vars = [h1, h2, h3, h4]

F = -sp.log( (sp.exp(-rho*h1) + sp.exp(-rho*h2) + sp.exp(-rho*h3) + sp.exp(-rho*h4) )/ N) / rho 

h_np = np.array([1, 2, 3, 4])
sub_pairs = {h1: h_np[0], h2: h_np[1], h3: h_np[2], h4: h_np[3]}

F_theory = float(F.subs(sub_pairs))

F_dh_theory = np.zeros(len(h_vars))
for i in range(len(h_vars)):
    symbol = F.diff(h_vars[i])
    F_dh_theory[i] = float(symbol.subs(sub_pairs))

F_dhdh_theory = np.zeros((len(h_vars),len(h_vars)))
for i in range(len(h_vars)):
    for j in range(len(h_vars)):
        symbol = F.diff(h_vars[i]).diff(h_vars[j])
        F_dhdh_theory[i,j] = float(symbol.subs(sub_pairs))

F_answer, F_dh_answer, F_dhdh_answer = doh.getSmoothMinimumAndLocalGradientAndHessian(rho, h_np)

print(np.allclose(F_theory, F_answer, rtol=1e-10, atol=1e-10))
print(np.allclose(F_dh_theory, F_dh_answer, rtol=1e-10, atol=1e-10))
print(np.allclose(F_dhdh_theory, F_dhdh_answer, rtol=1e-10, atol=1e-10))

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
x_vars = [x1, x2]

h1 = x1 + x2*x2
h2 = x1
h3 = x1*x1 + x2
h4 = x1*x2*x2
h_vars = [h1, h2, h3, h4]

rho = 5
N = 4
F = -sp.log( (sp.exp(-rho*h1) + sp.exp(-rho*h2) + sp.exp(-rho*h3) + sp.exp(-rho*h4) )/ N) / rho 
x_np = np.array([1, 2])
sub_pairs = {x1: x_np[0], x2: x_np[1]}

F_theory = float(F.subs(sub_pairs))

F_dx_theory = np.zeros(len(x_vars))
for i in range(len(x_vars)):
    symbol = F.diff(x_vars[i])
    F_dx_theory[i] = float(symbol.subs(sub_pairs))

F_dxdx_theory = np.zeros((len(x_vars),len(x_vars)))
for i in range(len(x_vars)):
    for j in range(len(x_vars)):
        symbol = F.diff(x_vars[i]).diff(x_vars[j])
        F_dxdx_theory[i,j] = float(symbol.subs(sub_pairs))

h_np = np.zeros(len(h_vars))
for i in range(len(h_vars)):
    symbol = h_vars[i]
    h_np[i] = float(symbol.subs(sub_pairs))

h_dx_np = np.zeros((len(h_vars),len(x_vars)))
for i in range(len(h_vars)):
    for j in range(len(x_vars)):
        symbol = h_vars[i].diff(x_vars[j])
        h_dx_np[i,j] = float(symbol.subs(sub_pairs))

h_dxdx_np = np.zeros((len(h_vars),len(x_vars),len(x_vars)))
for i in range(len(h_vars)):
    for j in range(len(x_vars)):
        for k in range(len(x_vars)):
            symbol = h_vars[i].diff(x_vars[j]).diff(x_vars[k])
            h_dxdx_np[i,j,k] = float(symbol.subs(sub_pairs))

F_answer, F_dx_answer, F_dxdx_answer = doh.getSmoothMinimumAndTotalGradientAndHessian(rho, h_np, h_dx_np, h_dxdx_np)
print(np.allclose(F_theory, F_answer, rtol=1e-10, atol=1e-10))
print(np.allclose(F_dx_theory, F_dx_answer, rtol=1e-10, atol=1e-10))
print(np.allclose(F_dxdx_theory, F_dxdx_answer, rtol=1e-10, atol=1e-10))