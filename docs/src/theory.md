# Radial Basis Functions Theory

Radial Basis Functions (RBF) use only a distance (typically Euclidean) when constructing the basis. For example, if we wish to build an interpolator we get the following linear combination of RBFs

$$u(\mathbf{x})=\sum_{i=1}^{N} \alpha_{i} \phi(|\mathbf{x}-\mathbf{x}_{i}|)$$

where $| \cdot |$ is a norm (we will use Euclidean from here on) and so $|\mathbf{x}-\mathbf{x}_{i}| = r$ is the Euclidean distance (although it can be any) and $N$ is the number of data points.

There are several types of RBFs to choose from, some with a tunable shape parameter, $\varepsilon$. Here are some popular ones:

| Type                 | Function                               |
| -------------------- | -------------------------------------- |
| Polyharmonic Spline  | $\phi(r) = r^n$ where $n=1,3,5,7,\cdots$                      |
| Multiquadric             |    $\phi(r)=\sqrt{ (r \varepsilon)^{2}+ 1 }$                                    |
| Inverse Multiquadric | $\phi(r) = 1 / \sqrt{(r \varepsilon)^2+1}$ |
| Gaussian             | $\phi(r) = e^{-(r \varepsilon)^2}$               |

## Augmenting with Monomials

The interpolant may be augmented with a polynomial as

$$
u(\mathbf{x})=\sum_{i=1}^{N} \alpha_{i} \phi(\lvert \mathbf{x}-\mathbf{x}_{i} \rvert) + \sum_{i=1}^{N_{p}} \gamma_{i} p_{i}(\mathbf{x})
$$

where $N_{p}=\binom{m+d}{m}$ is the number of monomials ($m$ is the monomial order and $d$ is the dimension of $\mathbf{x}$) and $p_{i}(\mathbf{x})$ is the monomial term, or:

$$
p_{i}(\mathbf{x})=q_{i}(\lvert \mathbf{x}-\mathbf{x}_{i} \rvert)
$$

where $q_{i}$ is the $i$-th monomial in $\mathbf{q}=\left[\begin{array}{c} 1, x, y, x^2, xy, y^2 \end{array}\right]$ in 2D, for example. By collocation the expansion of the augmented interpolant at all the nodes $\mathbf{x}_{i}$ where $i=1\cdots N$, there results a linear system for the interpolant weights as:

$$
\begin{pmatrix}
\mathbf{A} & \mathbf{P} \\\
\mathbf{P}^T & 0
\end{pmatrix}
\begin{pmatrix}
\boldsymbol{\alpha} \\\
\boldsymbol{\gamma}
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{u} \\\
0
\end{pmatrix}
$$

where

$$
\mathbf{A}=
\begin{pmatrix}
\phi(|\mathbf{x}_{1}-\mathbf{x}_{1}|) & \cdots & \phi(|\mathbf{x}_{1}-\mathbf{x}_{N}|) \\\
\vdots & \ddots & \vdots \\\
\phi(|\mathbf{x}_{N}-\mathbf{x}_{1}|) & \cdots & \phi(|\mathbf{x}_{N}-\mathbf{x}_{N}|)
\end{pmatrix}
\quad
\mathbf{P}=
\begin{pmatrix}
p_{1}(\mathbf{x}_{1}) & \cdots & p_{N_p}(\mathbf{x}_{1}) \\
\vdots & \ddots & \vdots \\\
p_{1}(\mathbf{x}_{N}) & \cdots & p_{N_p}(\mathbf{x}_{N})
\end{pmatrix}
$$

and $\mathbf{u}$ is the vector of dependent data points

$$
\mathbf{u}=
\begin{pmatrix}
u(\mathbf{x}_{1}) \\
\vdots \\
u(\mathbf{x}_{N})
\end{pmatrix}
$$

and $\boldsymbol{\alpha}$ and $\boldsymbol{\gamma}$ are the interpolation coefficients. Note that the equations relating to $\mathbf{P}^T$ are included to ensure optimal interpolation and unique solvability given that conditionally positive radial functions are used and the nodes in the subdomain form a unisolvent set. See (Fasshauer, et al. - Meshfree Approximation Methods with Matlab) and (Wendland, et al. - Scattered Data Approximation).

This augmentation of the system is highly encouraged for a couple main reasons:

1. Increased accuracy especially for flat fields and near boundaries
2. To ensure the linear system has a unique solution

See (Flyer, et al. - On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy) for more information on this.

## Local Collocation

The original RBF method employing the Kansa approach which connects all the nodes in the domain and, as such, is a _global_ method. Due to ill-conditioning and computational cost, this approach scales poorly; therefore, a _local_ approach is used instead. In the _local_ approach, each node is influenced only by its $k$ nearest neighbors which helps solve the issues related to _global_ collocation.

## Hermite Approach for Boundary Stencils

When a stencil is **centered around an internal node but includes boundary nodes**, standard RBF-FD using Kansa's approach can lead to ill-conditioning and singularity issues. This occurs because applying boundary operators $\mathcal{B}$ (such as normal derivatives for Neumann conditions) to the interpolation conditions breaks the symmetry of the local matrix $\mathbf{A}$.

**The RBF-HFD (Hermite Finite Difference) method resolves this issue by modifying the basis functions only for stencils near boundaries.** Instead of keeping the same basis regardless of boundary conditions and applying the operator to the interpolation conditions (which creates asymmetry), the Hermite approach modifies the basis itself.

For a stencil with $m_I$ internal nodes and $m_B$ boundary nodes ($m = m_I + m_B$), the approximate solution $u^h$ (the RBF interpolant) takes the form:

$$
u^h(\mathbf{x}_c) = \sum_{j=1}^{m_I} \alpha_j \phi(|\mathbf{x}_c - \mathbf{x}_j|) + \sum_{j=m_I+1}^{m} \alpha_j \mathcal{B}_2 \phi(|\mathbf{x}_c - \mathbf{x}_j|) + \sum_{k=1}^{N_p} \beta_k p_k(\mathbf{x}_c)
$$

where $\mathbf{x}_c$ is the stencil center (evaluation point), $\alpha_j$ are RBF coefficients, $\beta_k$ are polynomial coefficients, and $\mathcal{B}_2$ denotes the boundary operator applied to the second argument of the kernel (i.e., to $\mathbf{x}_j$), while $\mathcal{B}_1$ would denote application to the first argument (i.e., to $\mathbf{x}_c$). The key insight is that **the basis function is changed** from $\phi(\cdot, \mathbf{x}_j)$ to $\mathcal{B}_2 \phi(\cdot, \mathbf{x}_j)$ for boundary nodes.

The local system becomes:

$$
\begin{pmatrix}
\mathbf{A}_{I,I} & \mathcal{B}_2\mathbf{A}_{I,B} & \mathbf{P}_I \\
\mathcal{B}_1\mathbf{A}_{B,I} & \mathcal{B}_1\mathcal{B}_2\mathbf{A}_{B,B} & \mathcal{B}\mathbf{P}_B \\
\mathbf{P}_I^T & (\mathcal{B}\mathbf{P}_B)^T & 0
\end{pmatrix}
\begin{pmatrix}
\boldsymbol{\alpha}_I \\
\boldsymbol{\alpha}_B \\
\boldsymbol{\beta}
\end{pmatrix}
=
\begin{pmatrix}
\mathbf{u}_I \\
\mathbf{g} \\
0
\end{pmatrix}
$$

where subscripts $I$ and $B$ denote internal and boundary quantities, respectively. The matrix blocks $\mathbf{A}_{I,I}$, $\mathbf{A}_{I,B}$, $\mathbf{A}_{B,I}$, and $\mathbf{A}_{B,B}$ represent RBF evaluations between internal-internal, internal-boundary, boundary-internal, and boundary-boundary nodes. The vectors $\boldsymbol{\alpha}_I$ and $\boldsymbol{\alpha}_B$ are the RBF coefficients for internal and boundary nodes, $\boldsymbol{\beta}$ are the polynomial coefficients, $\mathbf{u}_I$ contains function values at internal nodes, and $\mathbf{g}$ contains boundary condition values. **This system is now symmetric and positive definite** (for appropriate RBF kernels), ensuring unique solvability regardless of the boundary condition type.

### Key Advantages

1. **Restores symmetry** of the local interpolation matrix for boundary stencils
2. **Eliminates singularity issues** that arise with differential boundary operators
3. **Minimal computational overhead** - no additional nodes or information required
4. **Flexible** - works with any linear boundary operator $\mathcal{B}$

The Hermite approach is **only applied to stencils that include boundary nodes**. For internal stencils far from boundaries, the standard RBF-FD formulation remains unchanged, maintaining computational efficiency where boundary effects are not present.

## Alternative Approach: Boundary Nodes as Unknowns

In some applications, particularly multi-region or coupled problems, it may be advantageous to solve the governing equation at boundary nodes as well, treating **all nodes (interior and boundary) as unknowns** in the global system. When this strategy is adopted, an alternative implementation to the Hermite approach becomes available.

Rather than modifying the basis functions for boundary nodes, this approach **maintains the standard RBF basis** $\{\phi(| \cdot - \mathbf{x}_j |)\}$ for all nodes regardless of their position. The key distinction is in how local systems are constructed:

- **When the stencil includes boundary nodes but the evaluation point is interior**: Apply the standard RBF-FD method unchanged. Boundary neighbors contribute as regular unknowns with no special treatment.

- **When the evaluation point itself is on the boundary**: Instead of modifying basis functions, modify the **right-hand side** of the local system. For a boundary point $\mathbf{x}_c$ (the stencil center) with operator $\mathcal{B}$, construct the RHS as $\mathcal{B}\boldsymbol{\phi}(\mathbf{x}_c)$ and $\mathcal{B}\mathbf{p}(\mathbf{x}_c)$ rather than using the standard differential operator.

This means the collocation matrix $\mathbf{A}$ always uses the standard kernel evaluation:

$$
[\mathbf{A}]_{ij} = \phi(|\mathbf{x}_i - \mathbf{x}_j|)
$$

maintaining symmetry trivially. The local system for a boundary evaluation point becomes:

$$
\begin{pmatrix}
\mathbf{A} & \mathbf{P} \\
\mathbf{P}^T & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf{w} \\
\boldsymbol{\lambda}
\end{pmatrix}
=
\begin{pmatrix}
\mathcal{B}\boldsymbol{\phi}(\mathbf{x}_c) \\
\mathcal{B}\mathbf{p}(\mathbf{x}_c)
\end{pmatrix}
$$

This approach is **significantly simpler than the Hermite method** because stencil classification depends only on the evaluation point type, the same RBF basis is used everywhere, and interior stencils with boundary neighbors require no special treatment. It is particularly suitable when boundary values are genuinely unknown and must be determined simultaneously with the interior solution, such as in fluid-structure interaction, multi-physics coupling, or domain decomposition methods.

## Constructing an Operator

In the Radial Basis Function - Finite Difference method (RBF-FD), a stencil is built to approximate derivatives using the same neighborhoods/subdomains of $N$ points. This is used in the [[MeshlessMultiphysics.jl]] package. For example, if $\mathcal{L}$ represents a linear differential operator, one can express the differentiation of the field variable $u$ at the center of the subdomain $\mathbf{x}_{c}$ in terms of some weights $\mathbf{w}$ and the field variable values on all the nodes within the subdomain as

$$
\mathcal{L}u(\mathbf{x}_{c}) = \sum_{i=1}^{N}w_{i}u(\mathbf{x}_{i})
$$

We can find $\mathbf{w}$ by satisfying

$$
\sum_{i=1}^{N}w_{i}\phi_{j}(\mathbf{x}_{i}) = \mathcal{L}\phi_{j}(\mathbf{x}_{c})
$$

for each basis function $\phi_{j}$ (where $\phi_j(\mathbf{x}_i) = \phi(|\mathbf{x}_i - \mathbf{x}_j|)$) and $j=1,\cdots, N$, and if you wish to augment with monomials, we also must satisfy

$$
\sum_{i=1}^{N_{p}}\lambda_{i}p_{j}(\mathbf{x}_{i}) = \mathcal{L}p_{j}(\mathbf{x}_{c})
$$

which leads to an overdetermined problem

$$
\mathrm{min} \left( \frac{1}{2} \mathbf{w}\mathbf{A}^{\intercal}\mathbf{w} - \mathbf{w}^{\intercal} \mathcal{L}\phi \right), \text{ subject to } \mathbf{P}^{\intercal}\mathbf{w}=\mathcal{L}\mathbf{p}
$$

which is practically solved as a linear system for the weights $\mathbf{w}$ as

$$
\begin{pmatrix}
\mathbf{A} & \mathbf{P} \\
\mathbf{P}^T & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf{w} \\
\boldsymbol{\lambda}
\end{pmatrix}
=
\begin{pmatrix}
\mathcal{L}\boldsymbol{\phi} \\
\mathcal{L}\mathbf{p}
\end{pmatrix}
$$

where $\boldsymbol{\lambda}$ are treated as Lagrange multipliers and are discarded after solving the linear system. The vectors are defined as

$$
\mathcal{L}\boldsymbol{\phi}=
\begin{pmatrix}
\mathcal{L}\boldsymbol{\phi}(|\mathbf{x}_{1}-\mathbf{x}_{c}|) \\
\vdots \\
\mathcal{L}\boldsymbol{\phi}(|\mathbf{x}_{N}-\mathbf{x}_{c}|)
\end{pmatrix}
\hspace{2em}
\mathcal{L}\mathbf{p}=
\begin{pmatrix}
\mathcal{L}p_{1}(\mathbf{x}_{c}) \\
\vdots \\
\mathcal{L}p_{N_{p}}(\mathbf{x}_{c})
\end{pmatrix}
$$

where $\mathcal{L}\boldsymbol{\phi}$ is the vector of the operator applied to each RBF basis function evaluated at the stencil nodes, and $\mathcal{L}\mathbf{p}$ is the vector of the operator applied to each polynomial basis function.

### Hermite Operator Construction

When constructing operators for stencils near boundaries using the Hermite approach, the system is modified to:

$$
\begin{pmatrix}
\mathbf{A}_{I,I} & \mathcal{B}_2\mathbf{A}_{I,B} & \mathbf{P}_I \\
\mathcal{B}_1\mathbf{A}_{B,I} & \mathcal{B}_1\mathcal{B}_2\mathbf{A}_{B,B} & \mathcal{B}\mathbf{P}_B \\
\mathbf{P}_I^T & (\mathcal{B}\mathbf{P}_B)^T & 0
\end{pmatrix}
\begin{pmatrix}
\mathbf{w}_I \\
\mathbf{w}_B \\
\boldsymbol{\lambda}
\end{pmatrix}
=
\begin{pmatrix}
\mathcal{L}_1\boldsymbol{\phi}(\mathbf{x}_c, \mathcal{X}_I) \\
\mathcal{L}_1\mathcal{B}_2\boldsymbol{\phi}(\mathbf{x}_c, \mathcal{X}_B) \\
\mathcal{L}\mathbf{p}(\mathbf{x}_c)
\end{pmatrix}
$$

where $\mathcal{L}_1$ denotes the differential operator applied to the first argument of the kernel, $\mathcal{X}_I$ is the set of internal nodes in the stencil, and $\mathcal{X}_B$ is the set of boundary nodes in the stencil. This yields weight vectors $\mathbf{w}_I$ and $\mathbf{w}_B$ that properly account for boundary conditions while maintaining symmetry. The global system assembly then proceeds as:

$$
\mathcal{L}u^h(\mathbf{x}_i) = \sum_{j \in \mathcal{X}_{i,I}} w_j(\mathbf{x}_i) u(\mathbf{x}_j) + \sum_{j \in \mathcal{X}_{i,B}} w_j(\mathbf{x}_i) g(\mathbf{x}_j)
$$

where $g(\mathbf{x}_j)$ represents the boundary condition values at boundary nodes.

## References

- Fasshauer, G. E., & McCourt, M. (2015). *Kernel-based Approximation Methods using MATLAB*. World Scientific. https://doi.org/10.1142/9335

- Flyer, N., Fornberg, B., Bayona, V., & Barnett, G. A. (2016). On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. *Journal of Computational Physics*, 321, 21-38. https://doi.org/10.1016/j.jcp.2016.05.026

- Shankar, V., Wright, G. B., Kirby, R. M., & Fogelson, A. L. (2015). A radial basis function (RBF)-finite difference (FD) method for diffusion and reaction-diffusion equations on surfaces. *Journal of Scientific Computing*, 63(3), 745-768. https://doi.org/10.1007/s10915-014-9914-1

- Wendland, H. (2004). *Scattered Data Approximation*. Cambridge University Press. https://doi.org/10.1017/CBO9780511617539

- Wright, G. B., & Fornberg, B. (2006). Scattered node compact finite difference-type formulas generated from radial basis functions. *Journal of Computational Physics*, 212(1), 99-123. https://doi.org/10.1016/j.jcp.2005.05.030
