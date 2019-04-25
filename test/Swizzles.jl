using Swizzles: mask

@testset "Swizzles" begin

  A = [1 2 3; 4 5 6; 7 8 9]

  @test mask(Swizzle(+, 1, 3, 5)) === (1, 3, 5)
  @test mask(typeof(Swizzle(+, 10, 8, 6, 4, 2))) === (10, 8, 6, 4, 2)

  @test_throws DimensionMismatch Pour(max, (1, nil)).(A, [1, 2, 3, 4]) == [3; 6; 9]

  C = zeros(Int, 4)

  C .= Sum().(A)
  @test C == [45; 45; 45; 45]

  @test Sum(()).(A) == A
  @test Sum(:).(A) == 45
  @test Reduce(+, ()).(A) == A
  @test Reduce(+, :).(A) == 45
  @test Sum().([]) == nothing
  @test Beam(2, 3).([]) == Array{Any}(undef,1,0)
  @test_throws DimensionMismatch Beam(nil, 3).([])
  @test_throws DimensionMismatch Beam(1, nil, nil).(Array{Any}(undef,1,0))
  @test Sum(2).(Array{Any}(undef,1,0)) == [nothing, nothing][1:1,1:1]

  @test Pour(max, (1, nil)).(A) == [3; 6; 9]
  @test Pour(min, (1, nil)).(A) == [1; 4; 7]
  @test Pour(max, (1,)).(A) == [3; 6; 9]
  @test Pour(min, (1,)).(A) == [1; 4; 7]
  @test Pour(max, (nil, 1)).(A) == [7; 8; 9]
  @test Pour(min, (nil, 1)).(A) == [1; 2; 3]
  @test Pour(max, (2, nil)).(A) == [3  6  9]
  @test Pour(min, (2, nil)).(A) == [1  4  7]
  @test Pour(max, (nil, 2)).(A) == [7  8  9]
  @test Pour(min, (nil, 2)).(A) == [1  2  3]

  R = [0; 0; 0;]
  @test (R .= Pour(max, (1, nil)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Pour(min, (1, nil)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Pour(max, (1,)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Pour(min, (1,)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Pour(max, (nil, 1)).(A)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test (R .= Pour(min, (nil, 1)).(A)) == [1; 2; 3]

  R = [0 0 0]
  @test (R .= Pour(max, (2, nil)).(A)) == [3  6  9]
  R = [0 0 0]
  @test (R .= Pour(min, (2, nil)).(A)) == [1  4  7]
  R = [0 0 0]
  @test (R .= Pour(max, (nil, 2)).(A)) == [7  8  9]
  R = [0 0 0]
  @test (R .= Pour(min, (nil, 2)).(A)) == [1  2  3]

  @test Pour(max, (1, nil, 2)).(A) == [3; 6; 9]
  @test Pour(min, (1, nil, 2)).(A) == [1; 4; 7]
  @test Pour(max, (nil, 1, 2)).(A) == [7; 8; 9]
  @test Pour(min, (nil, 1, 2)).(A) == [1; 2; 3]

  R = [0; 0; 0;]
  @test (R .= Pour(max, (1, nil, 2)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Pour(min, (1, nil, 2)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Pour(max, (nil, 1, 2)).(A)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test (R .= Pour(min, (nil, 1, 2)).(A)) == [1; 2; 3]

  R = [0; 0; 0;]
  @test_throws DimensionMismatch R .= Beam((nil, 1, 2)).(A)
  @test_throws DimensionMismatch R .= Beam((1, nil, 2)).(A)

  @test Beam((2, 1)).(A) == transpose(A)

  R = [0 0 0;
       0 0 0;
       0 0 0]
  @test (R .= Pour(min, (2, 1)).(A)) == transpose(A)

  @test Beam(2).([11; 12; 13]) == [11 12 13]
  @test Pour(max, ()).([11; 12; 13]) == 13
  @test Pour(max, ()).((11, 12, 13)) == 13

  R = [0 0 0]
  @test (R .= Beam((2,)).([11; 12; 13])) == [11 12 13]
  R = [0]
  @test Pour(max, ()).((11, 12, 13)) == 13

  @test Pour(+).(A) == 45
  @test Pour(+, 2).(A) == [6 15 24]
  @test Pour(+, 1).(A) == [6; 15; 24]
  @test Pour(+, 1,2).(A) == A
  @test Pour(+, 1,2,3).(A) == A
  @test Pour(+, 2,1).(A) == transpose(A)
  @test Pour(+, 2,1,3).(A) == transpose(A)

  @test_throws DimensionMismatch Beam().(A)
  @test_throws DimensionMismatch Beam(2).(A)
  @test_throws DimensionMismatch Beam(1).(A)
  @test Beam(1,2).(A) == A
  @test Beam(1,2,3).(A) == A
  @test Beam(2,1).(A) == transpose(A)
  @test Beam(2,1,3).(A) == transpose(A)

  @test Reduce(+).(A) == 45
  @test Reduce(+, 1).(A) == [12 15 18]
  @test Reduce(+, 2).(A) == [6 15 24]'
  @test Reduce(+, 1, 2).(A) == [45]'
  @test Reduce(+, 2, 1).(A) == [45]'
  @test Reduce(+, 2, 1, 3).(A) == [45]'

  @test Sum().(A) == 45
  @test Sum(1).(A) == [12 15 18]
  @test Sum(2).(A) == [6 15 24]'
  @test Sum(1, 2).(A) == [45]'
  @test Sum(2, 1).(A) == [45]'
  @test Sum(2, 1, 3).(A) == [45]'

  @test Drop(+).(A) == 45
  @test Drop(+, 1).(A) == [12, 15, 18]
  @test Drop(+, 2).(A) == [6; 15; 24]
  @test Drop(+, 1, 2).(A) == 45
  @test Drop(+, 2, 1).(A) == 45
  @test Drop(+, 2, 1, 3).(A) == 45

  @test SumOut().(A) == 45
  @test SumOut(1).(A) == [12; 15; 18]
  @test SumOut(2).(A) == [6; 15; 24]
  @test SumOut(1, 2).(A) == 45
  @test SumOut(2, 1).(A) == 45
  @test SumOut(2, 1, 3).(A) == 45

  @test_throws DimensionMismatch Focus().(A)
  @test_throws DimensionMismatch Focus(2).(A)
  @test_throws DimensionMismatch Focus(1).(A)
  @test Focus(1,2).(A) == A
  @test Focus(1,2,3).(A) == reshape(A, 3, 3, 1)
  @test Focus(2,1).(A) == transpose(A)
  @test Focus(2,1,3).(A) == reshape(transpose(A), 3, 3, 1)

  @test Swizzle(+).(A) == 45
  @test Swizzle(+, 1).(A) == [6; 15; 24]
  @test Swizzle(+, 2).(A) == [12; 15; 18]
  @test Swizzle(+, 1,2).(A) == A
  @test Swizzle(+, 1,2,3).(A) == reshape(A, 3, 3, 1)
  @test Swizzle(+, 2,1).(A) == transpose(A)
  @test Swizzle(+, 2,1,3).(A) == reshape(transpose(A), 3, 3, 1)

  @test Beam(1).((1, 2)) isa Tuple
  @test Pour(+, (1,)).((1, 2)) isa Tuple
  @test SumOut(1).((1.0, 2)) isa Float64

  A = rand(1,1)
  B = rand(1,1)

  @test SumOut(2).(A.*Beam(2,3).(B)) isa Matrix
  @test Sum().(A.*Beam(2,3).(B)) isa Float64
  @test SumOut(2).(A.*Beam(2,3).(B)) == A * B

  A = rand(5,7)
  B = rand(7,6)

  @test SumOut(2).(A.*Beam(2,3).(B)) ≈ A * B

  @test reshape(Beam(2, 4).(A).*Beam(1, 3).(B), size(A, 1) * size(B, 1), :) == kron(A, B)

  x = rand(3)
  y = rand(4)

  @test reshape(Beam(2).(x).*y, :) == kron(x, y)

  z = Array{Int64,0}(undef)
  z[] = 13

  @test Beam(3).(z) == 13

  A = [1 2 3 4 5]
  Focus(nil, nil, 2).(A) == Beam(nil, 3).(A)

  @test Swizzle(+).(rand(3,3)) isa Float64
  @test copy(Swizzle(+)(rand(3,3))) isa AbstractArray{Float64, 0}

  A = reshape(1:27, 3, 3, 3)

  @test Sum(2).(A)[1, 1:1, 2:3] == Sum(2)(A)[1, 1:1, 2:3]
  @test Sum(2).(A)[1, 1:1, 3] == Sum(2)(A)[1, 1:1, 3]
  @test Sum(2).(A)[1, 1, 3:3] == Sum(2)(A)[1, 1, 3:3]


end
