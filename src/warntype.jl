using Swizzle
A = B = rand(3,3)
foo(A, B) = Sum(2).(A.*Beam(2, 3).(B))
using InteractiveUtils
@show @code_lowered(foo(A, B))
@show @code_warntype(foo(A, B))
@show @code_llvm(foo(A, B))
