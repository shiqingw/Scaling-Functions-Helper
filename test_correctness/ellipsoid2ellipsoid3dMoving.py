import numpy as np
import scalingFunctionsHelperPy as doh
from scipy.spatial.transform import Rotation

Q1 = np.random.rand(3)**2 + 1
Q1 = np.diag(Q1)
mu1 = np.zeros(3)

Q2 = np.random.rand(3)**2 + 1
Q2 = np.diag(Q2)
mu2 = np.random.rand(3)

q1 = np.random.rand(4)
q1 = q1 / np.linalg.norm(q1)
q1 = q1/np.sign(q1[-1])
R1 = Rotation.from_quat(q1).as_matrix()
d1 = np.random.rand(3)

q2 = np.random.rand(4)
q2 = q2 / np.linalg.norm(q2)
q2 = q2/np.sign(q2[-1])
R2 = Rotation.from_quat(q2).as_matrix()
d2 = np.zeros(3)

SF1_moving = doh.Ellipsoid3d(True, Q1, mu1)
SF1_not_moving = doh.Ellipsoid3d(False, Q1, mu1)

SF2_moving = doh.Ellipsoid3d(True, Q2, mu2)
SF2_not_moving = doh.Ellipsoid3d(False, Q2, mu2)

p_rimon1 = doh.rimonMethod3d(SF1_moving, d1, q1, SF2_moving, d2, q2)
alpha1, alpha_dx1, alpha_dxdx1 = doh.getGradientAndHessian3d(p_rimon1, SF1_moving, d1, q1, SF1_moving, d2, q2)
print(alpha1)
print(alpha_dx1)
print(alpha_dxdx1 - alpha_dxdx1.T)

# alpha2, p_rimon2, alpha_dx2_tmp, alpha_dxdx2_tmp = doh1.getGradientAndHessianEllipsoids(mu1_w, q1, Q1, R1, Q2_w, mu2_w)
# alpha_dx2 = np.zeros_like(alpha_dx2_tmp)
# alpha_dx2[:3] = alpha_dx2_tmp[4:7]
# alpha_dx2[3:] = alpha_dx2_tmp[:4]

# alpha_dxdx2 = np.zeros_like(alpha_dxdx2_tmp)
# alpha_dxdx2[:3,:3] = alpha_dxdx2_tmp[4:7,4:7]
# alpha_dxdx2[3:,:3] = alpha_dxdx2_tmp[:4,4:7]
# alpha_dxdx2[:3,3:] = alpha_dxdx2_tmp[4:7,:4]
# alpha_dxdx2[3:,3:] = alpha_dxdx2_tmp[:4,:4]

# print(np.allclose(p_rimon1, p_rimon2))
# print(np.allclose(alpha1, alpha2))
# print(np.allclose(alpha_dx1, alpha_dx2))
# print(np.allclose(alpha_dxdx1, alpha_dxdx2))

# import timeit
# print(timeit.timeit(lambda: doh2.rimonMethod3d(SF1, d1, q1, SF2, d2, q2), number=10000))
# print(timeit.timeit(lambda: doh2.getGradientAndHessian3d(p_rimon1, SF1, d1, q1, SF2, d2, q2), number=10000))
