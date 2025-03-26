_num_ops(_) = 1
_num_ops(ℒ::Tuple) = length(ℒ)
_prepare_b(_, T, n) = zeros(T, n)
_prepare_b(ℒ::Tuple, T, n) = zeros(T, n, length(ℒ))
