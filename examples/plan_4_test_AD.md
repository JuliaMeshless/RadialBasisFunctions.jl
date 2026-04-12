I wrote this test for AD, and I havent written it yet, but I want to make sure that the test is well-defined and covers all the necessary aspects of the moonkake implementation for AD. First read it and make sure it makes sense, if it does not make sense, please ask me to clarify. If it does make sense, then you can proceed to implement the test.

Write a test for the moonkake implementation for AD.

The test should do this:
- write a set of points in a 2D space, these points should be laid out in a grid pattern and slightly perturbed by some random noise.
- also define the adjacency list (say with a simple 14-nearest neighbor connectivity) for these points, using find_neighbors(points, 14).
- calculate the weights for the Laplacian operator based on the adjacency list and the distances between the points. (use PHS(3; poly_deg=2) to construct everything)

- Define a simple function that can be solved exactly on these points using a quadratic polynomial, for example: f(x, y) = ax^2 + by^2 + cxy + dx + ey + f, where a, b, c, d, e, and f are constants.

NOTE: The choice of a quadratic test function with PHS(3; poly_deg=2) is intentional. The RBF scheme with quadratic polynomial augmentation reproduces quadratics EXACTLY, so the Laplacian residual (W * f_exact - ∇²f_exact) will be machine-precision zero at all interior points. This is a FEATURE, not a bug: the purpose of this test is to validate the AD implementation, NOT to test the RBF scheme's approximation accuracy. The RBF scheme is assumed correct and is the starting point. Using exact reproduction means any nonzero gradient of the loss w.r.t. point positions comes purely from the geometry sensitivity of the weights, not from approximation error — this is exactly what we want to test.

- Construct the Laplace equation using the calculated weights (these are provided by _build_weights) and use the defined function as the right-hand side of the equation. Also use the values of the function at the points as the boundary conditions. We should be landing on a system that is Ax=b, where A is the matrix of weights, x is the unknown function values at the points, and b is the right-hand side derived from the defined function. Everything should be pretty compact therefore you should be able to solve this system using a direct solver.

  System assembly follows the Macchiato pattern:
  - A[interior rows, :] = W_laplacian[interior, :]   (from _build_weights)
  - A[boundary rows, :] = I                           (Dirichlet rows: zero row, set diagonal to 1)
  - b[interior] = ∇²f_exact(x_interior)              (known analytically for the quadratic)
  - b[boundary] = f_exact(x_boundary)
  The BC overwrite rows are simple index operations and do not need to be differentiated through.

- Compare the solution obtained from solving the system with the exact values of the function at the points, and assert that they are close within a reasonable tolerance. Do not continue if these are different.

- Now define a loss function that can be computed starting from the exact solution and the computed solution.

- Now we use the Mooncake implementation for AD to compute the gradients of this loss function with respect to the positions of the points. This will involve backpropagating through the entire process of constructing the weights, solving the system, and computing the loss.

  Implementation note: point positions must be passed as a flat Vector{Float64} to the loss
  function (since that is what DifferentiationInterface expects), and reconstructed as
  Vector{SVector{2,Float64}} inside the closure. See the _build_weights tests in
  test/extensions/autodiff_di.jl (around line 213) for the pattern:
      pts_vec = [SVector{2}(pts[2i-1], pts[2i]) for i in 1:N]

- Finally, we can check the computed gradients against numerical gradients obtained by perturbing the positions of the points and observing the change in the loss function. We should assert that the computed gradients are close to the numerical gradients within a reasonable tolerance.

