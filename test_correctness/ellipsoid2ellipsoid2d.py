import sympy as sp
import numpy as np
import diffOptHelper as doh1
import scalingFunctionsHelperPy as doh2
from scipy.spatial.transform import Rotation

Q1 = np.random.rand(2)**2 + 1
Q1 = np.diag(Q1)
Q2 = np.random.rand(2)**2 + 1
Q2 = np.diag(Q2)
mu1 = np.zeros(2)
mu2 = np.random.rand(2)
SF1 = doh2.Ellipsoid2d(True, Q1, mu1)
SF2 = doh2.Ellipsoid2d(False, Q2, mu2)

theta1 = np.random.rand()
R1 = np.array([[np.cos(theta1), -np.sin(theta1)],
               [np.sin(theta1), np.cos(theta1)]])
d1 = np.random.rand(2)
Q1_w = R1 @ Q1 @ R1.T
mu1_w = R1 @ mu1 + d1
Q1_w_ans = SF1.getWorldQuadraticCoefficient(theta1)
mu1_w_ans = SF1.getWorldCenter(d1, theta1)

print(np.allclose(Q1_w, Q1_w_ans))
print(np.allclose(mu1_w, mu1_w_ans))

theta2 = np.random.rand()
R2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                [np.sin(theta2), np.cos(theta2)]])
d2 = np.random.rand(2)
Q2_w = R2 @ Q2 @ R2.T
mu2_w = R2 @ mu2 + d2

p_rimon1 = doh2.rimonMethod(Q1_w, mu1_w, Q2_w, mu2_w)
p_rimon1_2 = doh2.rimonMethod2d(SF1, d1, theta1, SF2, d2, theta2)
alpha1, alpha_dx1, alpha_dxdx1 = doh2.getGradientAndHessian2d(p_rimon1, SF1, d1, theta1, SF2, d2, theta2)
alpha1_2, alpha_dx1_2 = doh2.getGradient2d(p_rimon1, SF1, d1, theta1, SF2, d2, theta2)
print(np.allclose(p_rimon1, p_rimon1_2))
print(np.allclose(alpha1, alpha1_2))
print(np.allclose(alpha_dx1, alpha_dx1_2))

alpha2, p_rimon2, alpha_dx2_tmp, alpha_dxdx2_tmp = doh1.getGradientAndHessianEllipses(mu1_w, theta1, Q1, R1, Q2_w, mu2_w)
alpha_dx2 = np.zeros_like(alpha_dx2_tmp)
alpha_dx2[:2] = alpha_dx2_tmp[1:]
alpha_dx2[2:] = alpha_dx2_tmp[:1]

alpha_dxdx2 = np.zeros_like(alpha_dxdx2_tmp)
alpha_dxdx2[:2, :2] = alpha_dxdx2_tmp[1:, 1:]
alpha_dxdx2[2:, 2:] = alpha_dxdx2_tmp[:1, :1]
alpha_dxdx2[:2, 2:] = alpha_dxdx2_tmp[1:, :1]
alpha_dxdx2[2:, :2] = alpha_dxdx2_tmp[:1, 1:]

print(np.allclose(p_rimon1, p_rimon2))
print(np.allclose(alpha1, alpha2))
print(np.allclose(alpha_dx1, alpha_dx2))
print(np.allclose(alpha_dxdx1, alpha_dxdx2))
