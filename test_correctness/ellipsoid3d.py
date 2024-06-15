import sympy as sp
import numpy as np
import diffOptHelper2 as doh
from scipy.spatial.transform import Rotation

qx, qy, qz, qw = sp.symbols('qx qy qz qw', real=True)
r11, r12, r13, r21, r22, r23, r31, r32, r33 = sp.symbols('r11 r12 r13 r21 r22 r23 r31 r32 r33', real=True)
dx, dy, dz = sp.symbols('dx dy dz', real=True)
px, py, pz = sp.symbols('px py pz', real=True)

x_vars = [dx, dy, dz, qx, qy, qz, qw]
p_vars = [px, py, pz]

R = sp.Matrix([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
p = sp.Matrix([px, py, pz])
d = sp.Matrix([dx, dy, dz])
P = R.T @ (p-d)

Q_np = np.random.rand(3,3)
Q_np = Q_np + Q_np.T
mu_np = np.random.rand(3)

rpy_np = np.random.rand(3)
R_np = Rotation.from_euler('zyx', rpy_np).as_matrix()
q_np = Rotation.from_euler('zyx', rpy_np).as_quat()
p_np = np.random.rand(3)
d_np = np.random.rand(3)
P_np = R_np.T @ (p_np-d_np)
sub_pairs = {qx:q_np[0], qy:q_np[1], qz:q_np[2], qw:q_np[3],
             px:p_np[0], py:p_np[1], pz:p_np[2],
             dx:d_np[0], dy:d_np[1], dz:d_np[2]}

SF = doh.Ellipsoid3d(True, Q_np, mu_np)
F = ((P-mu_np[:,np.newaxis]).T @ Q_np @ (P-mu_np[:,np.newaxis]))[0]

dim_P, dim_p, dim_x = 3, 3, 7
# # F_dP
# theory = 2 * Q_np @ (P_np-mu_np)
# answer = SF.getBodyFdP(P_np)
# print(theory - answer)

# # F_dPdP
# theory = 2 * Q_np
# answer = SF.getBodyFdPdP(P_np)
# print(theory - answer)

# # F_dPdPdP
# theory = np.zeros((dim_P,dim_P,dim_P))
# answer = SF.getBodyFdPdPdP(P_np)
# print(theory - answer)

# # F_dp
# theory = np.zeros((dim_p))
# for i in range(dim_p):
#     symbol = F.diff(p_vars[i])
#     theory[i] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdp(p_np, d_np, q_np, R_np)
# print(theory - answer)

# # F_dx
# theory = np.zeros((dim_x))
# for i in range(dim_x):
#     symbol = F.diff(x_vars[i])
#     theory[i] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdx(p_np, d_np, q_np, R_np)
# print(theory - answer)

# # F_dpdp
# theory = np.zeros((dim_p,dim_p))
# for i in range(dim_p):
#     for j in range(dim_p):
#         symbol = F.diff(p_vars[i]).diff(p_vars[j])
#         theory[i,j] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdpdp(p_np, d_np, q_np, R_np)
# print(theory - answer)

# # F_dpdx
# theory = np.zeros((dim_p,dim_x))
# for i in range(dim_p):
#     for j in range(dim_x):
#         symbol = F.diff(p_vars[i]).diff(x_vars[j])
#         theory[i,j] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdpdx(p_np, d_np, q_np, R_np)
# print(theory - answer)

# F_dxdx
# theory = np.zeros((dim_x,dim_x))
# for i in range(dim_x):
#     for j in range(dim_x):
#         symbol = F.diff(x_vars[i]).diff(x_vars[j])
#         theory[i,j] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdxdx(p_np, d_np, q_np, R_np)
# print(theory - answer)

# F_dpdpdp
# theory = np.zeros((dim_p,dim_p,dim_p))
# for i in range(dim_p):
#     for j in range(dim_p):
#         for k in range(dim_p):
#             symbol = F.diff(p_vars[i]).diff(p_vars[j]).diff(p_vars[k])
#             theory[i,j,k] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdpdpdp(p_np, d_np, q_np, R_np)
# print(theory - answer)

# F_dpdpdx
# theory = np.zeros((dim_p,dim_p,dim_x))
# for i in range(dim_p):
#     for j in range(dim_p):
#         for k in range(dim_x):
#             symbol = F.diff(p_vars[i]).diff(p_vars[j]).diff(x_vars[k])
#             theory[i,j,k] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdpdpdx(p_np, d_np, q_np, R_np)
# print(theory - answer)

# F_dpdxdx
# theory = np.zeros((dim_p,dim_x,dim_x))
# for i in range(dim_p):
#     for j in range(dim_x):
#         for k in range(dim_x):
#             symbol = F.diff(p_vars[i]).diff(x_vars[j]).diff(x_vars[k])
#             theory[i,j,k] = symbol.subs(sub_pairs)
# answer = SF.getWorldFdpdxdx(p_np, d_np, q_np, R_np)
# print(theory - answer)
