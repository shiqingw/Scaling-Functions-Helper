import sympy as sp
import numpy as np
import diffOptHelper2 as doh
from scipy.spatial.transform import Rotation

theta = sp.symbols('theta', real=True)
dx, dy = sp.symbols('dx dy', real=True)
px, py = sp.symbols('px py', real=True)

x_vars = [dx, dy, theta]
p_vars = [px, py]

R = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                [sp.sin(theta), sp.cos(theta)]])
p = sp.Matrix([px, py])
d = sp.Matrix([dx, dy])
P = R.T @ (p-d)

dim_P = 2
dim_p = len(p_vars)
dim_x = len(x_vars)

a_np = np.random.rand(dim_p)
b_np = np.random.rand(1)

theta_np = np.random.rand()
R_np = np.array([[np.cos(theta_np), -np.sin(theta_np)],
                    [np.sin(theta_np), np.cos(theta_np)]])
p_np = np.random.rand(dim_p)
d_np = np.random.rand(dim_p)
P_np = R_np.T @ (p_np-d_np)
sub_pairs = {theta:theta_np,
             px:p_np[0], py:p_np[1],
             dx:d_np[0], dy:d_np[1]}

SF = doh.Hyperplane2d(True, a_np, b_np)
F = (a_np[:,np.newaxis].T @ P + b_np[:,np.newaxis])[0]

dim_P, dim_p, dim_x = 2, 2, 3

# F(P)
theory = float(F.subs(sub_pairs))
answer = SF.getBodyF(P_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F(p)
theory = float(F.subs(sub_pairs))
answer = SF.getWorldF(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dp
theory = np.zeros((dim_p))
for i in range(dim_p):
    symbol = F.diff(p_vars[i])
    theory[i] = symbol.subs(sub_pairs)
answer = SF.getWorldFdp(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dx
theory = np.zeros((dim_x))
for i in range(dim_x):
    symbol = F.diff(x_vars[i])
    theory[i] = symbol.subs(sub_pairs)
answer = SF.getWorldFdx(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dpdp
theory = np.zeros((dim_p,dim_p))
for i in range(dim_p):
    for j in range(dim_p):
        symbol = F.diff(p_vars[i]).diff(p_vars[j])
        theory[i,j] = symbol.subs(sub_pairs)
answer = SF.getWorldFdpdp(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dpdx
theory = np.zeros((dim_p,dim_x))
for i in range(dim_p):
    for j in range(dim_x):
        symbol = F.diff(p_vars[i]).diff(x_vars[j])
        theory[i,j] = symbol.subs(sub_pairs)
answer = SF.getWorldFdpdx(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dxdx
theory = np.zeros((dim_x,dim_x))
for i in range(dim_x):
    for j in range(dim_x):
        symbol = F.diff(x_vars[i]).diff(x_vars[j])
        theory[i,j] = symbol.subs(sub_pairs)
answer = SF.getWorldFdxdx(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dpdpdp
theory = np.zeros((dim_p,dim_p,dim_p))
for i in range(dim_p):
    for j in range(dim_p):
        for k in range(dim_p):
            symbol = F.diff(p_vars[i]).diff(p_vars[j]).diff(p_vars[k])
            theory[i,j,k] = symbol.subs(sub_pairs)
answer = SF.getWorldFdpdpdp(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dpdpdx
theory = np.zeros((dim_p,dim_p,dim_x))
for i in range(dim_p):
    for j in range(dim_p):
        for k in range(dim_x):
            symbol = F.diff(p_vars[i]).diff(p_vars[j]).diff(x_vars[k])
            theory[i,j,k] = symbol.subs(sub_pairs)
answer = SF.getWorldFdpdpdx(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

# F_dpdxdx
theory = np.zeros((dim_p,dim_x,dim_x))
for i in range(dim_p):
    for j in range(dim_x):
        for k in range(dim_x):
            symbol = F.diff(p_vars[i]).diff(x_vars[j]).diff(x_vars[k])
            theory[i,j,k] = symbol.subs(sub_pairs)
answer = SF.getWorldFdpdxdx(p_np, d_np, theta_np)
print(np.allclose(theory, answer, rtol=1e-10, atol=1e-10))

import timeit
print(timeit.timeit('SF.getWorldFdpdxdx(p_np, d_np, theta_np)', globals=globals(), number=10000))