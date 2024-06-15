import numpy as np
import diffOptHelper2 as doh

# a = np.array([1., 2., 3.])
# b = np.array([4., 5., 6.])

# print(doh.getDualVariable(a,b))

# h = np.array([1., 2., 3.])
# rho = 10.0
# F, F_dh, F_dhdh = doh.getSmoothMinimumAndLocalGradientAndHessian(rho, h)
# h_dx = np.random.rand(len(h),6)
# h_dxdx = np.random.rand(len(h),6,6)
# h_dxdx = np.transpose(h_dxdx, (0,2,1)) + h_dxdx

# F, F_dx, F_dxdx = doh.getSmoothMinimumAndTotalGradientAndHessian(rho, h, h_dx, h_dxdx)
# print(F_dx - F_dh @ h_dx)
# print(F_dxdx - (np.einsum('i,ijk->jk', F_dh, h_dxdx) + h_dx.T @ F_dhdh @ h_dx))

Q = np.eye(3)
P = np.array([3., 2., 1.])
mu = np.array([1., 2., 3.])
SF = doh.Ellipsoid3d(True, Q, mu)
print(SF.isMoving)
SF.isMoving = False
print(SF.isMoving)
# print(SF.getBodyFdPdPdP(P))
# print(SF.getBodyPdpdxdx())