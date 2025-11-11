"""
Shared utility functions for tests in RadialBasisFunctions.jl
"""

"""
    rmse(test, correct)

Compute root mean square error relative to the norm of the correct values.
"""
rmse(test, correct) = sqrt(sum((test - correct) .^ 2) / sum(correct .^ 2))

"""
    mean_percent_error(test, correct)

Compute mean percent error between test and correct values.
"""
mean_percent_error(test, correct) = mean(abs.((test .- correct) ./ correct)) * 100

# Export utility functions
export rmse, mean_percent_error
