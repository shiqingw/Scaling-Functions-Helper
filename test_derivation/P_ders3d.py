import sympy as sp
from sympy.printing import cxxcode

qx, qy, qz, qw = sp.symbols('qx qy qz qw', real=True)
dx, dy, dz = sp.symbols('dx dy dz', real=True)
px, py, pz = sp.symbols('px py pz', real=True)

x_vars = [dx, dy, dz, qx, qy, qz, qw]
p_vars = [px, py, pz]

# R = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
#                 [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
#                 [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
R = sp.Matrix([[1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]])
p = sp.Matrix([px, py, pz])
d = sp.Matrix([dx, dy, dz])
P = R.T @ (p-d)

dim_P = 3
dim_p = len(p_vars)
dim_x = len(x_vars)

# R
# for i in range(dim_p):
#     for j in range(dim_p):
#         result = R[i,j]
#         if result != 0:
#             print(f'R({i},{j}) = {cxxcode(result)};')


# # P_dp
# P_dp = sp.simplify(sp.simplify(P.jacobian(p_vars))
# print(P_dp)

# P_dx
# for i in range(dim_P):
#     for j in range(3, dim_x):
#         result = P[i].diff(x_vars[j])
#         if result != 0:
#             print(f'P_dx({i},{j}) = {result};')


# P_dpdx
# for i in range(dim_P):
#     for j in range(dim_p):
#         for k in range(dim_x):
#             result = P[i].diff(p_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dpdx({i},{j},{k}) = {result};')

# P_dxdx
# for i in range(dim_P):
#     for j in range(dim_x):
#         for k in range(dim_x):
#             result = P[i].diff(x_vars[j]).diff(x_vars[k])
#             if result != 0:
#                 print(f'P_dxdx({i},{j},{k}) = {result};')

# # P_dpdxdx
for i in range(dim_P):
    for j in range(dim_p):
        for k in range(dim_x):
            for l in range(dim_x):
                result = P[i].diff(p_vars[j]).diff(x_vars[k]).diff(x_vars[l])
                if result != 0:
                    print(f'P_dpdxdx({i},{j},{k},{l}) = {result};')