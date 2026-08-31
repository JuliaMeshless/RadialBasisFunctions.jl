# Radial Basis Functions Theory

Radial Basis Functions (RBF) use only a distance (typically Euclidean) when constructing the basis. For example, an interpolator for $u(\mathbf{x})$ is given by the linear combination of RBFs

$$u(\mathbf{x})=\sum_{i=1}^{N} \alpha_{i} \phi(\lvert \mathbf{x}-\mathbf{x}_{i} \rvert)$$

where $\mid \cdot \mid$ is a norm (we will use Euclidean from here on) and so $\lvert \mathbf{x}-\mathbf{x}_{i} \rvert = r$ is the Euclidean distance (although any norm can be used) and $N$ is the number of data points.

There are several types of RBFs to choose from, some with a tunable shape parameter, $\varepsilon$. Here are some popular ones:

| Type                 | Function                               |
| -------------------- | -------------------------------------- |
| Polyharmonic Spline  | $\phi(r) = r^n$ where $n \in \{1,3,5,7\}$                      |
| Inverse Multiquadric | $\phi(r) = 1 / \sqrt{(r \varepsilon)^2+1}$ |
| Gaussian             | $\phi(r) = e^{-(r \varepsilon)^2}$               |

## Augmenting with Monomials

The interpolant may be augmented with a polynomial:

```math
u(\mathbf{x})=\sum_{i=1}^{N} \alpha_{i} \phi(\lvert \mathbf{x}-\mathbf{x}_{i} \rvert) + \sum_{i=1}^{N_{p}} \gamma_{i} p_{i}(\mathbf{x})
```

where $N_{p}=\begin{pmatrix} m+d \\ m \end{pmatrix}$ is the number of monomials ($m$ is the monomial order and $d$ is the dimension of $\mathbf{x}$) and $p_{i}(\mathbf{x})$ is the monomial term. 

For instance, in 2D with $m=2$, we have $N_{p}=6$ and
```math
p_{i}(\mathbf{x}) \in \left\{ 1, x, y, x^2, xy, y^2 \right\}
```

When we require the interpolation to be exact on a set of data points $\{\mathbf{x}_{i}, u(\mathbf{x}_{i})\}$ where $i=1,\cdots,N$, we obtain the following linear system for the interpolation coefficients:

```math
\left[\begin{array}{cc}
\mathbf{A} & \mathbf{P} \\
\mathbf{P}^\mathrm{T} & 0
\end{array}\right]
\left[\begin{array}{c}
\boldsymbol{\alpha} \\
\boldsymbol{\gamma}
\end{array}\right]=
\left[\begin{array}{c}
\mathbf{u} \\
0
\end{array}\right]
```

where

```math
\mathbf{A}=
\left[\begin{array}{ccc}
\phi(\lvert \mathbf{x}_{1}-\mathbf{x}_{1} \rvert) & \cdots & \phi(\lvert \mathbf{x}_{1}-\mathbf{x}_{N} \rvert) \\
\vdots & & \vdots \\
\phi(\lvert \mathbf{x}_{N}-\mathbf{x}_{1} \rvert) & \cdots & \phi(\lvert \mathbf{x}_{N}-\mathbf{x}_{N} \rvert)
\end{array}\right]
\hspace{2em}
\mathbf{P}=
\left[\begin{array}{ccc}
p_{1}(\mathbf{x}_{1}) & \cdots & p_{N}(\mathbf{x}_{1}) \\
\vdots & & \vdots \\
p_{1}(\mathbf{x}_{N}) & \cdots & p_{N}(\mathbf{x}_{N})
\end{array}\right]
```

and $\mathbf{u}$ is the vector of dependent data points

```math
\mathbf{u}=
\left[\begin{array}{c}
u(\mathbf{x}_{1}) \\
\vdots \\
u(\mathbf{x}_{N})
\end{array}\right]
```

and $\boldsymbol{\alpha}$ and $\boldsymbol{\gamma}$ are the interpolation coefficients. Note that the equations relating to $\mathbf{P}^\mathrm{T}$ are included to ensure optimal interpolation and unique solvability, given that conditionally positive radial functions are used and the nodes in the subdomain form a unisolvent set. See (Fasshauer, et al. - Meshfree Approximation Methods with Matlab) and (Wendland, et al. - Scattered Data Approximation).

Polynomial augmentation of the system has two benefits:

1. Increases accuracy, especially for polynomial fields and near boundaries.
2. Ensures the linear system has a unique solution for certain types of RBFs (conditionally positive definite).

See (Flyer, et al. - On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy) for more information on this.

## Local Collocation

The traditional Kansa approach used in most RBF methods is based on constructing a unique interpolant for all the nodes in the domain. This involves coupling all nodes in the domain simultaneously and therefore makes it a _global_ method. Such a global approach, while theoretically exact, scales poorly: the resulting dense system becomes prohibitively expensive and increasingly ill-conditioned as the number of nodes grows, particularly in 3D, due to the curse of dimensionality. Instead, RadialBasisFunctions.jl employs a _local_ approach, where each node is influenced only by its $k$ nearest neighbors.

## Constructing an Operator

In the Radial Basis Function - Finite Difference method (RBF-FD), a stencil is built to approximate derivatives using the same neighborhoods/subdomains of $N$ points.
 <!-- This is used in the [[MeshlessMultiphysics.jl]] package. -->
For example, if $\mathcal{L}$ represents a linear differential operator, one can express the differentiation of the field variable $u$ at the center of the subdomain $\mathbf{x}_{c}$ in terms of some weights $\mathbf{w}$ and the field variable values on all the nodes within the subdomain as

```math
\mathcal{L}u(\mathbf{x}_{c}) = \sum_{i=1}^{N}w_{i}u(\mathbf{x}_{i})
```

We can find $\mathbf{w}$ by satisfying

```math
\sum_{i=1}^{N}w_{i}\phi_{j}(\mathbf{x}_{i}) = \mathcal{L}\phi_{j}(\mathbf{x}_{c})
```

for each basis function $\phi_{j}$ (where $\phi_j(\mathbf{x}_i) = \phi(\lvert \mathbf{x}_i - \mathbf{x}_j \rvert)$) and $j=1,\cdots, N$, and if you wish to augment with monomials, we also must satisfy

```math
\sum_{i=1}^{N_{p}}\lambda_{i}p_{j}(\mathbf{x}_{i}) = \mathcal{L}p_{j}(\mathbf{x}_{c})
```

which leads to an overdetermined problem

```math
\mathrm{min} \left( \frac{1}{2} \mathbf{w}\mathbf{A}^{\intercal}\mathbf{w} - \mathbf{w}^{\intercal} \mathcal{L}\phi \right), \text{ subject to } \mathbf{P}^{\intercal}\mathbf{w}=\mathcal{L}\mathbf{p}
```

which is practically solved as a linear system for the weights $\mathbf{w}$ as

```math
\left[\begin{array}{cc}
\mathbf{A} & \mathbf{P} \\
\mathbf{P}^\mathrm{T} & 0
\end{array}\right]
\left[\begin{array}{c}
\mathbf{w} \\
\boldsymbol{\lambda}
\end{array}\right]=
\left[\begin{array}{c}
\mathcal{L}\boldsymbol{\phi} \\
\mathcal{L}\mathbf{p}
\end{array}\right]
```

where $\boldsymbol{\lambda}$ are treated as Lagrange multipliers and are discarded after solving the linear system. The vectors are defined as

```math
\mathcal{L}\boldsymbol{\phi}=
\left[\begin{array}{c}
\mathcal{L}\boldsymbol{\phi}(\lvert \mathbf{x}_{1}-\mathbf{x}_{c} \rvert) \\
\vdots \\
\mathcal{L}\boldsymbol{\phi}(\lvert \mathbf{x}_{N}-\mathbf{x}_{c} \rvert)
\end{array}\right]
\hspace{2em}
\mathcal{L}\mathbf{p}=
\left[\begin{array}{c}
\mathcal{L}p_{1}(\mathbf{x}_{c}) \\
\vdots \\
\mathcal{L}p_{N_{p}}(\mathbf{x}_{c})
\end{array}\right]
```

where $\mathcal{L}\boldsymbol{\phi}$ is the vector of the operator applied to each RBF basis function evaluated at the stencil nodes, and $\mathcal{L}\mathbf{p}$ is the vector of the operator applied to each polynomial basis function.

## References

- Fasshauer, G. E., & McCourt, M. (2015). *Kernel-based Approximation Methods using MATLAB*. World Scientific. https://doi.org/10.1142/9335

- Flyer, N., Fornberg, B., Bayona, V., & Barnett, G. A. (2016). On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. *Journal of Computational Physics*, 321, 21-38. https://doi.org/10.1016/j.jcp.2016.05.026

- Shankar, V., Wright, G. B., Kirby, R. M., & Fogelson, A. L. (2015). A radial basis function (RBF)-finite difference (FD) method for diffusion and reaction-diffusion equations on surfaces. *Journal of Scientific Computing*, 63(3), 745-768. https://doi.org/10.1007/s10915-014-9914-1

- Wendland, H. (2004). *Scattered Data Approximation*. Cambridge University Press. https://doi.org/10.1017/CBO9780511617539

- Wright, G. B., & Fornberg, B. (2006). Scattered node compact finite difference-type formulas generated from radial basis functions. *Journal of Computational Physics*, 212(1), 99-123. https://doi.org/10.1016/j.jcp.2005.05.030
