import sympy as sp
import numpy as np
import diffOptHelper as doh1
import scalingFunctionsHelper as doh2
from scipy.spatial.transform import Rotation

Q1 = np.random.rand(3)**2 + 1
Q1 = np.diag(Q1)
Q2 = np.random.rand(3)**2 + 1
Q2 = np.diag(Q2)
mu1 = np.zeros(3)
mu2 = np.random.rand(3)
SF1 = doh2.Ellipsoid3d(True, Q1, mu1)
SF2 = doh2.Ellipsoid3d(False, Q2, mu2)

q1 = np.random.rand(4)
q1 = q1 / np.linalg.norm(q1)
q1 = q1/np.sign(q1[-1])
R1 = Rotation.from_quat(q1).as_matrix()
d1 = np.random.rand(3)
Q1_w = R1 @ Q1 @ R1.T
mu1_w = R1 @ mu1 + d1
Q1_w_ans = SF1.getWorldQuadraticCoefficient(q1)
mu1_w_ans = SF1.getWorldCenter(d1, q1)

print(np.allclose(Q1_w, Q1_w_ans))
print(np.allclose(mu1_w, mu1_w_ans))

q2 = np.array([0,0,0,1])
R2 = Rotation.from_quat(q2).as_matrix()
d2 = np.zeros(3)
Q2_w = R2 @ Q2 @ R2.T
mu2_w = R2 @ mu2 + d2

p_rimon1 = doh2.rimonMethod(Q1_w, mu1_w, Q2_w, mu2_w)
p_rimon1_2 = doh2.rimonMethod3d(SF1, d1, q1, SF2, d2, q2)
alpha1, alpha_dx1, alpha_dxdx1 = doh2.getGradientAndHessian3d(p_rimon1, SF1, d1, q1, SF2, d2, q2)
alpha1_2, alpha_dx1_2 = doh2.getGradient3d(p_rimon1, SF1, d1, q1, SF2, d2, q2)
print(np.allclose(p_rimon1, p_rimon1_2))
print(np.allclose(alpha1, alpha1_2))
print(np.allclose(alpha_dx1, alpha_dx1_2))

alpha2, p_rimon2, alpha_dx2_tmp, alpha_dxdx2_tmp = doh1.getGradientAndHessianEllipsoids(mu1_w, q1, Q1, R1, Q2_w, mu2_w)
alpha_dx2 = np.zeros_like(alpha_dx2_tmp)
alpha_dx2[:3] = alpha_dx2_tmp[4:7]
alpha_dx2[3:] = alpha_dx2_tmp[:4]

alpha_dxdx2 = np.zeros_like(alpha_dxdx2_tmp)
alpha_dxdx2[:3,:3] = alpha_dxdx2_tmp[4:7,4:7]
alpha_dxdx2[3:,:3] = alpha_dxdx2_tmp[:4,4:7]
alpha_dxdx2[:3,3:] = alpha_dxdx2_tmp[4:7,:4]
alpha_dxdx2[3:,3:] = alpha_dxdx2_tmp[:4,:4]

print(np.allclose(p_rimon1, p_rimon2))
print(np.allclose(alpha1, alpha2))
print(np.allclose(alpha_dx1, alpha_dx2))
print(np.allclose(alpha_dxdx1, alpha_dxdx2))
