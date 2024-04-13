import numpy as np
from fealpy.decorator import cartesian # 函数的输入变量的类型
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarLaplaceIntegrator      # (\nabla u, \nabla v)
from fealpy.fem import LinearForm
from fealpy.fem import ScalarSourceIntegrator         # (f, v)
from fealpy.fem import DirichletBC
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class MyData:
    def __init__(self, kappa=1.0):
        self.kappa = kappa # Robin 条件中的系数

    def domain(self):
        """
        @brief 模型定义域
        """
        return np.array([0, 1, 0, 1])

    @cartesian
    def solution(self, p):
        """
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        """
        @brief 源项
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val

    @cartesian
    def gradient(self, p):
        """
        @brief 真解梯度
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - 1.0) < 1e-12) | (np.abs( y -  0.0) < 1e-12)



# 建立 PDE 对象
pde = MyData()
# 定义域
domain = np.array([0, 1, 0, 1])
# 三角网格
mesh = TriangleMesh.from_box(domain, nx=10, ny=10)
p = 2 # 空间次数
space = LagrangeFESpace(mesh, p=p)

bform = BilinearForm(space)
# (\nabla u, \nabla v)
bform.add_domain_integrator(ScalarLaplaceIntegrator(q=p+2))

A = bform.assembly()
lform = LinearForm(space)
# (f, v)
si = ScalarSourceIntegrator(pde.source, q=p+2)
lform.add_domain_integrator(si)

F = lform.assembly()
# Dirichlet 边界条件
bc = DirichletBC(space,
      pde.dirichlet, threshold=pde.is_dirichlet_boundary)
uh = space.function()
A, F = bc.apply(A, F, uh)
uh[:] = spsolve(A, F)

L2Error = mesh.error(pde.solution, uh, q=p+2)
H1Error = mesh.error(pde.gradient, uh.grad_value, q=p+2)
print("L2error is", L2Error)
print("H1error is", H1Error)

uh.add_plot(plt)
plt.show()