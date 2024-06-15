import sympy as sp
from sympy.printing import cxxcode

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

# P_dp
# P_dp = sp.simplify(sp.simplify(P.jacobian(p_vars)))
# print(P_dp)

# P_dx
# for i in range(dim_P):
#     for j in range(dim_x):
#         result = P[i].diff(x_vars[j])
#         if result != 0:
#             print(f'P_dx({i},{j}) = {cxxcode(result)};')


# P_dpdx
# for i in range(dim_P):
#     for j in range(dim_p):
#         for k in range(dim_x):
#             result = P[i].diff(p_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dpdx({i},{j},{k}) = {cxxcode(result)};')

# P_dxdx
# for i in range(dim_P):
#     for j in range(dim_x):
#         for k in range(dim_x):
#             result = P[i].diff(x_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dxdx({i},{j},{k}) = {cxxcode(result)};')

# # P_dpdxdx
for i in range(dim_P):
    for j in range(dim_p):
        for k in range(dim_x):
            for l in range(dim_x):
                result = P[i].diff(p_vars[j]).diff(x_vars[k]).diff(x_vars[l])
                if result != 0:
                    print(f'P_dpdxdx({i},{j},{k},{l}) = {cxxcode(result)};')