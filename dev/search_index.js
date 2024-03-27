var documenterSearchIndex = {"docs":
[{"location":"api/#Exported-Functions","page":"API","title":"Exported Functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [RadialBasisFunctions]\nPrivate = false\nOrder   = [:function, :type]","category":"page"},{"location":"api/#RadialBasisFunctions.PHS-Union{Tuple{}, Tuple{T}} where T<:Int64","page":"API","title":"RadialBasisFunctions.PHS","text":"function PHS(n::T=3; poly_deg::T=2) where {T<:Int}\n\nConvienience contructor for polyharmonic splines.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.directional-Union{Tuple{T}, Tuple{B}, Tuple{D}, Tuple{AbstractVector{D}, AbstractVector{D}, AbstractVector}, Tuple{AbstractVector{D}, AbstractVector{D}, AbstractVector, B}} where {D<:AbstractArray, B<:AbstractRadialBasis, T<:Int64}","page":"API","title":"RadialBasisFunctions.directional","text":"function directional(data, eval_points, v, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the directional derivative, Directional.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.directional-Union{Tuple{T}, Tuple{B}, Tuple{D}, Tuple{AbstractVector{D}, AbstractVector}, Tuple{AbstractVector{D}, AbstractVector, B}} where {D<:AbstractArray, B<:AbstractRadialBasis, T<:Int64}","page":"API","title":"RadialBasisFunctions.directional","text":"function directional(data, v, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the directional derivative, Directional.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.gradient-Union{Tuple{AbstractVector{D}}, Tuple{T}, Tuple{B}, Tuple{D}, Tuple{AbstractVector{D}, B}} where {D<:AbstractArray, B<:AbstractRadialBasis, T<:Int64}","page":"API","title":"RadialBasisFunctions.gradient","text":"function gradient(data, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the gradient, Gradient.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.gradient-Union{Tuple{T}, Tuple{B}, Tuple{D}, Tuple{AbstractVector{D}, AbstractVector{D}}, Tuple{AbstractVector{D}, AbstractVector{D}, B}} where {D<:AbstractArray, B<:AbstractRadialBasis, T<:Int64}","page":"API","title":"RadialBasisFunctions.gradient","text":"function gradient(data, eval_points, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the gradient, Gradient. The resulting operator will only evaluate at eval_points.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.partial-Union{Tuple{B}, Tuple{T}, Tuple{D}, Tuple{AbstractVector{D}, AbstractVector{D}, T, T}, Tuple{AbstractVector{D}, AbstractVector{D}, T, T, B}} where {D<:AbstractArray, T<:Int64, B<:AbstractRadialBasis}","page":"API","title":"RadialBasisFunctions.partial","text":"function partial(data, eval_points, order, dim, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the partial derivative, Partial. The resulting operator will only evaluate at eval_points.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.partial-Union{Tuple{B}, Tuple{T}, Tuple{D}, Tuple{AbstractVector{D}, T, T}, Tuple{AbstractVector{D}, T, T, B}} where {D<:AbstractArray, T<:Int64, B<:AbstractRadialBasis}","page":"API","title":"RadialBasisFunctions.partial","text":"function partial(data, order, dim, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is the partial derivative, Partial, of order with respect to dim.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.regrid-Union{Tuple{B}, Tuple{T}, Tuple{D}, Tuple{AbstractVector{D}, AbstractVector{D}}, Tuple{AbstractVector{D}, AbstractVector{D}, B}} where {D<:AbstractArray, T<:Int64, B<:AbstractRadialBasis}","page":"API","title":"RadialBasisFunctions.regrid","text":"function regrid(data, eval_points, order, dim, basis; k=autoselect_k(data, basis))\n\nBuilds a RadialBasisOperator where the operator is an regrid from one set of points to another, data -> eval_points.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.AbstractPHS","page":"API","title":"RadialBasisFunctions.AbstractPHS","text":"abstract type AbstractPHS <: AbstractRadialBasis\n\nSupertype of all Polyharmonic Splines.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.AbstractRadialBasis","page":"API","title":"RadialBasisFunctions.AbstractRadialBasis","text":"abstract type AbstractRadialBasis end\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Directional","page":"API","title":"RadialBasisFunctions.Directional","text":"Directional <: VectorValuedOperator\n\nOperator for the directional derivative, or the inner product of the gradient and a direction vector.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Gaussian","page":"API","title":"RadialBasisFunctions.Gaussian","text":"struct Gaussian{T,D<:Int} <: AbstractRadialBasis\n\nGaussian radial basis function:ϕ(r) = e^-(ε r)^2\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Gradient","page":"API","title":"RadialBasisFunctions.Gradient","text":"Gradient <: VectorValuedOperator\n\nBuilds an operator for the gradient of a function.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Interpolator","page":"API","title":"RadialBasisFunctions.Interpolator","text":"struct Interpolator\n\nConstruct a radial basis interpolation.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Interpolator-Union{Tuple{B}, Tuple{Any, Any}, Tuple{Any, Any, B}} where B<:AbstractRadialBasis","page":"API","title":"RadialBasisFunctions.Interpolator","text":"function Interpolator(x, y, basis::B=PHS())\n\nConstruct a radial basis interpolator.\n\n\n\n\n\n","category":"method"},{"location":"api/#RadialBasisFunctions.Laplacian","page":"API","title":"RadialBasisFunctions.Laplacian","text":"Laplacian <: ScalarValuedOperator\n\nOperator for the sum of the second derivatives w.r.t. each independent variable.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.MonomialBasis","page":"API","title":"RadialBasisFunctions.MonomialBasis","text":"struct MonomialBasis{T<:Int,B<:Function}\n\nMultivariate Monomial basis. n ∈ N: length of array, i.e., x ∈ Rⁿ deg ∈ N: degree\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.PHS1","page":"API","title":"RadialBasisFunctions.PHS1","text":"struct PHS1{T<:Int} <: AbstractPHS\n\nPolyharmonic spline radial basis function:ϕ(r) = r\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.PHS3","page":"API","title":"RadialBasisFunctions.PHS3","text":"struct PHS3{T<:Int} <: AbstractPHS\n\nPolyharmonic spline radial basis function:ϕ(r) = r^3\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.PHS5","page":"API","title":"RadialBasisFunctions.PHS5","text":"struct PHS5{T<:Int} <: AbstractPHS\n\nPolyharmonic spline radial basis function:ϕ(r) = r^5\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.PHS7","page":"API","title":"RadialBasisFunctions.PHS7","text":"struct PHS7{T<:Int} <: AbstractPHS\n\nPolyharmonic spline radial basis function:ϕ(r) = r^7\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Partial","page":"API","title":"RadialBasisFunctions.Partial","text":"Partial <: ScalarValuedOperator\n\nBuilds an operator for a first order partial derivative.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.RadialBasisOperator","page":"API","title":"RadialBasisFunctions.RadialBasisOperator","text":"struct RadialBasisOperator\n\nOperator of data using a radial basis with potential monomial augmentation.\n\n\n\n\n\n","category":"type"},{"location":"api/#RadialBasisFunctions.Regrid","page":"API","title":"RadialBasisFunctions.Regrid","text":"Regrid\n\nBuilds an operator for interpolating from one set of points to another.\n\n\n\n\n\n","category":"type"},{"location":"api/#Private","page":"API","title":"Private","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [RadialBasisFunctions]\nPublic = false\nOrder   = [:function, :type]","category":"page"},{"location":"api/#RadialBasisFunctions.autoselect_k-Union{Tuple{B}, Tuple{Vector, B}} where B<:AbstractRadialBasis","page":"API","title":"RadialBasisFunctions.autoselect_k","text":"autoselect_k(data::Vector, basis<:AbstractRadialBasis)\n\nSee Bayona, 2017 - https://doi.org/10.1016/j.jcp.2016.12.008\n\n\n\n\n\n","category":"method"},{"location":"theory/#Radial-Basis-Functions-Theory","page":"Theory","title":"Radial Basis Functions Theory","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"Radial Basis Functions (RBF) use only a distance (typically Euclidean) when constructing the basis. For example, if we wish to build an interpolator we get the following linear combination of RBFs","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(mathbfx)=sum_i=1^N alpha_i phi(lvert mathbfx-mathbfx_i rvert)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"where mid cdot mid is a norm (we will use Euclidean from here on) and so lvert mathbfx-mathbfx_i rvert = r is the Euclidean distance (although it can be any) and N is the number of data points.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"There are several types of RBFs to choose from, some with a tunable shape parameter, varepsilon. Here are some popular ones:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Type Function\nPolyharmonic Spline phi(r) = r^n where n=1357dots\nMultiquadric phi(r)=sqrt (r varepsilon)^2+ 1 \nInverse Multiquadric phi(r) = 1  sqrt(r varepsilon)^2+1\nGaussian phi(r) = e^-(r varepsilon)^2","category":"page"},{"location":"theory/#Augmenting-with-Monomials","page":"Theory","title":"Augmenting with Monomials","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The interpolant may be augmented with a polynomial as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(mathbfx)=sum_i=1^N alpha_i phi(lvert mathbfx-mathbfx_i rvert) + sum_i=1^N_p gamma_i p_i(mathbfx)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"where N_p=beginpmatrix m+d  m endpmatrix is the number of monomials (m is the monomial order and d is the dimension of mathbfx) and p_i(mathbfx) is the monomial term, or:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"p_i(mathbfx)=q_i(lvert mathbfx-mathbfx_i rvert)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"where q_i is the i-th monomial in mathbfq=beginbmatrix 1 x y x^2 xy y^2 endbmatrix in 2D, for example. By collocation the expansion of the augmented interpolant at all the nodes mathbfx_i where i=1dots N, there results a linear system for the interpolant weights as:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"beginbmatrix\nmathbfA  mathbfP \nmathbfP^mathrmT  0\nendbmatrix\nbeginbmatrix\nboldsymbolalpha \nboldsymbolgamma\nendbmatrix=\nbeginbmatrix\nmathbff \n0\nendbmatrix","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"where","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"mathbfA=\nbeginbmatrix\nphi(lvert mathbfx_1-mathbfx_1 rvert)  dots  phi(lvert mathbfx_1-mathbfx_N rvert) \nvdots   vdots \nphi(lvert mathbfx_N-mathbfx_1 rvert)  dots  phi(lvert mathbfx_N-mathbfx_N rvert)\nendbmatrix\nhspace2em\nmathbfp=\nbeginbmatrix\np_1(mathbfx_1)  dots  p_N(mathbfx_1) \nvdots   vdots \np_1(mathbfx_N)  dots  p_N(mathbfx_N)\nendbmatrix","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"and mathbff is the vector of dependent data points","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"mathbff=\nbeginbmatrix\nf(mathbfx_1) \nvdots \nf(mathbfx_N)\nendbmatrix","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"and boldsymbolalpha and boldsymbolgamma are the interpolation coefficients. Note that the equations relating to mathbfP^mathrmT are included to ensure optimal interpolation and unique solvability given that conditionally positive radial functions are used and the nodes in the subdomain form a unisolvent set. See (Fasshauer, et al. - Meshfree Approximation Methods with Matlab) and (Wendland, et al. - Scattered Data Approximation).","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"This augmentation of the system is highly encouraged for a couple main reasons:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Increased accuracy especially for flat fields and near boundaries\nTo ensure the linear system has a unique solution","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"See (Flyer, et al. - On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy) for more information on this.","category":"page"},{"location":"theory/#Local-Collocation","page":"Theory","title":"Local Collocation","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"The original RBF method employing the Kansa approach which connects all the nodes in the domain and, as such, is a global method. Due to ill-conditioning and computational cost, this approach scales poorly; therefore, a local approach is used instead. In the local approach, each node is influenced only by its k nearest neighbors which helps solve the issues related to global collocation.","category":"page"},{"location":"theory/#Constructing-an-Operator","page":"Theory","title":"Constructing an Operator","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"In the Radial Basis Function - Finite Difference method (RBF-FD), a stencil is built to approximate derivatives using the same neighborhoods/subdomains of N points. This is used in the [[MeshlessMultiphysics.jl]] package. For example, if mathcalL represents a linear differential operator, one can express the differentiation of the field variable u at the center of the subdomain mathbfx_c in terms of some weights mathbfw and the field variable values on all the nodes within the subdomain as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"mathcalLu(mathbfx_c) = sum_i=1^Nw_iu(mathbfx_i)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"We can find mathbfw by satisfying","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"sum_i=1^Nw_iphi_j(mathbfx_i) = mathcalLphi_j(mathbfx_c)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"for each phi_j where j=1dots N and if you wish to augment with monomials, we also must satisfy","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"sum_i=1^N_plambda_ip_j(mathbfx_i) = mathcalLp_j(mathbfx_c)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"which leads to an overdetermined problem","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"mathrmmin left( frac12 mathbfwmathbfA^intercalmathbfw - mathbfw^intercal mathcalLphi right) text subject to  mathbfP^intercalmathbfw=mathcalLmathbfp","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"which is practically solved as a linear system for the weights mathbfw as","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"beginbmatrixmathbfA  mathbfP \nmathbfP^mathrmT  0\nendbmatrix\nbeginbmatrix\nmathbfw \nboldsymbollambda\nendbmatrix=\nbeginbmatrix\nmathcalLboldsymbolphi \nmathcalLmathbfp\nendbmatrix","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"where boldsymbollambda are treated as Lagrange multipliers and are discarded after solving the linear system and","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"mathcalLboldsymbolphi=\nbeginbmatrix\nmathcalLboldsymbolphi(lvert mathbfx_1-mathbfx_c rvert) \nvdots \nmathcalLboldsymbolphi(lvert mathbfx_N-mathbfx_c rvert)\nendbmatrix\nhspace2em\nmathcalLmathbfp=\nbeginbmatrix\nmathcalLp_1(mathbfx_c) \nvdots \nmathcalLp_N_p(mathbfx_c)\nendbmatrix","category":"page"},{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"First, let's load the package along with the StaticArrays.jl package which we use for each data point","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"using RadialBasisFunctions\nusing StaticArrays","category":"page"},{"location":"getting_started/#Interpolation","page":"Getting Started","title":"Interpolation","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Suppose we have a set of data mathbfx where mathbfx_i in mathbbR^2, and we want to interpolate a function fmathbbR^2 rightarrow mathbbR","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"f(x) = 2*x[1]^2 + 3*x[2]\nx = [SVector{2}(rand(2)) for _ in 1:1000]\ny = f.(x)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"and now we can build the interpolator","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"interp = Interpolator(x, y)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"and evaluate it at a new point","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"x_new = [rand(2) for _ in 1:5]\ny_new = interp(x_new)\ny_true = f.(x_new)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"and compare the error","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"abs.(y_true .- y_new)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Wow! The error is numerically zero! Well... we set ourselves up for success here. Interpolator (along with RadialBasisOperator) has an optional argument to provide the type of radial basis including the degree of polynomial augmentation. The default basis is a cubic polyharmonic spline with 2nd degree polynomial augmentation (which the constructor is PHS(3, poly_deg=2)) and given the underlying function we are interpolating is a 2nd order polynomial itself, we are able to represent it exactly (up to machine precision). Let's see what happens when we only use 1st order polynomial augmentation","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"interp = Interpolator(x, y, PHS(3, poly_deg=1))\ny_new = interp(x_new)\nabs.(y_true .- y_new)","category":"page"},{"location":"getting_started/#Operators","page":"Getting Started","title":"Operators","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"This package also provides an API for operators. There is support for several built-in operators along with support for user-defined operators. Currently, we have implementations for","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"partial derivative (1st and 2nd order)\ngradient\nlaplacian","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"but we plan to add more in the future. Please make and issue or pull request for additional operators.","category":"page"},{"location":"getting_started/#Partial-Derivative","page":"Getting Started","title":"Partial Derivative","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We can take the same data as above and build a partial derivative operator with a similar construction as the interpolator. For the partial we need to specify the order of differentiation we want along with the dimension for which to take the partial. We can also supply some optional arguments such as the basis and number of points in the stencil. The function inputs order is partial(x, order, dim, basis; k=5)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"df_x_rbf = partial(x, 1, 1)\n\n# define exact\ndf_x(x) = 4*x[1]\n\n# error\nall(abs.(df_x.(x) .- df_x_rbf(y)) .< 1e-10)","category":"page"},{"location":"getting_started/#Laplacian","page":"Getting Started","title":"Laplacian","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Building a laplacian operator is as easy as","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"lap_rbf = laplacian(x)\n\n# define exact\nlap(x) = 4\n\n# error\nall(abs.(lap.(x) .- lap_rbf(y)) .< 1e-8)","category":"page"},{"location":"getting_started/#Gradient","page":"Getting Started","title":"Gradient","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We can also retrieve the gradient. This is really just a convenience wrapper around Partial.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"grad = gradient(x)\n\n# define exacts\ndf_x(x) = 4*x[1]\ndf_y(x) = 3\n\n# error\nall(df_x.(x) .≈ grad(y)[1])","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"all(df_y.(x) .≈ grad(y)[2])","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = RadialBasisFunctions","category":"page"},{"location":"#RadialBasisFunctions.jl","page":"Home","title":"RadialBasisFunctions.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for RadialBasisFunctions.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package intends to provide tools for all things regarding Radial Basis Functions (RBF). ","category":"page"},{"location":"","page":"Home","title":"Home","text":"Feature Status\nInterpolation ✅\nRegridding ✅\nPartial derivative (partial f) ✅\nLaplacian (nabla^2 f, Delta f) ✅\nGradient (nabla f) ✅\nDirectional Derivative (nabla f cdot v) ✅\nCustom / user supplied (mathcalL f) ✅\ndivergence (textrmdiv mathbfF or nabla cdot mathbfF) ❌\ncurl (nabla times mathbfF) ❌\nReduced Order Models (i.e. POD) ❌","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently, we support the following types of RBFs (all have polynomial augmentation by default, but is optional)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Type Function, phi(r)\nPolyharmonic Spline r^n where n=1357\nInverse Multiquadric 1  sqrt(r varepsilon)^2+1\nGaussian e^-(r varepsilon)^2","category":"page"},{"location":"","page":"Home","title":"Home","text":"where varepsilon is a user-supplied shape parameter.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Simply install the latest stable release using Julia's package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add RadialBasisFunctions","category":"page"},{"location":"#Current-Limitations","page":"Home","title":"Current Limitations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A critical dependency of this package is NearestNeighbors.jl which requires that the dimension of each data point is inferrable. To quote from NearestNeighbors.jl:\nThe data, i.e., the points to build up the tree from. It can either bea matrix of size nd × np with the points to insert in the tree where nd is the dimensionality of the points and np is the number of points\na vector of vectors with fixed dimensionality, nd, which must be part of the type. Specifically, data should be a Vector{V}, where V is itself a subtype of an AbstractVector and such that eltype(V) and length(V) are defined. (For example, with 3D points, V = SVector{3, Float64} works because eltype(V) = Float64 and length(V) = 3 are defined in V.)\nThat said, we currently only support the second option here (Vector{AbstractVector}), but plan to support matrix inputs in the future.\nInterpolator uses all points, but there are plans to support local collocation / subdomains like the operators use.","category":"page"}]
}
