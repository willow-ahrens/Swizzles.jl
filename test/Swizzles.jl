@testset "Swizzles" begin

  A = [1 2 3; 4 5 6; 7 8 9]

  @test_throws DimensionMismatch Beam(max, (1, nil)).(A, [1, 2, 3, 4]) == [3; 6; 9]

  @test Sum(()).(A) == A
  @test Sum(:).(A) == 45
  @test Reduce(+, ()).(A) == A
  @test Reduce(+, :).(A) == 45
  @test Sum().([]) == nothing
  @test Yeet(2, 3).([]) == Array{Any}(undef,1,0)
  @test_throws DimensionMismatch Yeet(nil, 3).([])
  @test_throws DimensionMismatch Yeet(1, nil, nil).(Array{Any}(undef,1,0))
  @test Sum(2).(Array{Any}(undef,1,0)) == [nothing, nothing][1:1,1:1]

  @test Beam(max, (1, nil)).(A) == [3; 6; 9]
  @test Beam(min, (1, nil)).(A) == [1; 4; 7]
  @test Beam(max, (1,)).(A) == [3; 6; 9]
  @test Beam(min, (1,)).(A) == [1; 4; 7]
  @test Beam(max, (nil, 1)).(A) == [7; 8; 9]
  @test Beam(min, (nil, 1)).(A) == [1; 2; 3]
  @test Beam(max, (2, nil)).(A) == [3  6  9]
  @test Beam(min, (2, nil)).(A) == [1  4  7]
  @test Beam(max, (nil, 2)).(A) == [7  8  9]
  @test Beam(min, (nil, 2)).(A) == [1  2  3]

  R = [0; 0; 0;]
  @test (R .= Beam(max, (1, nil)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Beam(min, (1, nil)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Beam(max, (1,)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Beam(min, (1,)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Beam(max, (nil, 1)).(A)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test (R .= Beam(min, (nil, 1)).(A)) == [1; 2; 3]

  R = [0 0 0]
  @test (R .= Beam(max, (2, nil)).(A)) == [3  6  9]
  R = [0 0 0]
  @test (R .= Beam(min, (2, nil)).(A)) == [1  4  7]
  R = [0 0 0]
  @test (R .= Beam(max, (nil, 2)).(A)) == [7  8  9]
  R = [0 0 0]
  @test (R .= Beam(min, (nil, 2)).(A)) == [1  2  3]

  @test Beam(max, (1, nil, 2)).(A) == [3; 6; 9]
  @test Beam(min, (1, nil, 2)).(A) == [1; 4; 7]
  @test Beam(max, (nil, 1, 2)).(A) == [7; 8; 9]
  @test Beam(min, (nil, 1, 2)).(A) == [1; 2; 3]

  R = [0; 0; 0;]
  @test (R .= Beam(max, (1, nil, 2)).(A)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test (R .= Beam(min, (1, nil, 2)).(A)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test (R .= Beam(max, (nil, 1, 2)).(A)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test (R .= Beam(min, (nil, 1, 2)).(A)) == [1; 2; 3]

  R = [0; 0; 0;]
  @test_throws DimensionMismatch R .= Yeet((nil, 1, 2)).(A)
  @test_throws DimensionMismatch R .= Yeet((1, nil, 2)).(A)

  @test Yeet((2, 1)).(A) == transpose(A)

  R = [0 0 0;
       0 0 0;
       0 0 0]
  @test (R .= Beam(min, (2, 1)).(A)) == transpose(A)

  @test Yeet(2).([11; 12; 13]) == [11 12 13]
  @test Beam(max, ()).([11; 12; 13]) == 13
  @test Beam(max, ()).((11, 12, 13)) == 13

  R = [0 0 0]
  @test (R .= Yeet((2,)).([11; 12; 13])) == [11 12 13]
  R = [0]
  @test Beam(max, ()).((11, 12, 13)) == 13

  @test Beam(+).(A) == 45
  @test Beam(+, 2).(A) == [6 15 24]
  @test Beam(+, 1).(A) == [6; 15; 24]
  @test Beam(+, 1,2).(A) == A
  @test Beam(+, 1,2,3).(A) == A
  @test Beam(+, 2,1).(A) == transpose(A)
  @test Beam(+, 2,1,3).(A) == transpose(A)

  @test_throws DimensionMismatch Yeet().(A)
  @test_throws DimensionMismatch Yeet(2).(A)
  @test_throws DimensionMismatch Yeet(1).(A)
  @test Yeet(1,2).(A) == A
  @test Yeet(1,2,3).(A) == A
  @test Yeet(2,1).(A) == transpose(A)
  @test Yeet(2,1,3).(A) == transpose(A)

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

  @test DropSum().(A) == 45
  @test DropSum(1).(A) == [12; 15; 18]
  @test DropSum(2).(A) == [6; 15; 24]
  @test DropSum(1, 2).(A) == 45
  @test DropSum(2, 1).(A) == 45
  @test DropSum(2, 1, 3).(A) == 45

  @test_throws DimensionMismatch Yoink().(A)
  @test_throws DimensionMismatch Yoink(2).(A)
  @test_throws DimensionMismatch Yoink(1).(A)
  @test Yoink(1,2).(A) == A
  @test Yoink(1,2,3).(A) == reshape(A, 3, 3, 1)
  @test Yoink(2,1).(A) == transpose(A)
  @test Yoink(2,1,3).(A) == reshape(transpose(A), 3, 3, 1)

  @test Swizzle(+).(A) == 45
  @test Swizzle(+, 1).(A) == [6; 15; 24]
  @test Swizzle(+, 2).(A) == [12; 15; 18]
  @test Swizzle(+, 1,2).(A) == A
  @test Swizzle(+, 1,2,3).(A) == reshape(A, 3, 3, 1)
  @test Swizzle(+, 2,1).(A) == transpose(A)
  @test Swizzle(+, 2,1,3).(A) == reshape(transpose(A), 3, 3, 1)

  @test Yeet(1).((1, 2)) isa Tuple
  @test Beam(+, (1,)).((1, 2)) isa Tuple
  @test DropSum(1).((1.0, 2)) isa Float64

  A = rand(1,1)
  B = rand(1,1)

  @test DropSum(2).(A.*Yeet(2,3).(B)) isa Matrix
  @test Sum().(A.*Yeet(2,3).(B)) isa Float64
  @test DropSum(2).(A.*Yeet(2,3).(B)) == A * B

  A = rand(5,7)
  B = rand(7,6)

  @test DropSum(2).(A.*Yeet(2,3).(B)) ≈ A * B

  @test reshape(Yeet(2, 4).(A).*Yeet(1, 3).(B), size(A, 1) * size(B, 1), :) == kron(A, B)

  x = rand(3)
  y = rand(4)

  @test reshape(Yeet(2).(x).*y, :) == kron(x, y)

  z = Array{Int64,0}(undef)
  z[] = 13

  @test Yeet(3).(z) == 13

  A = [1 2 3 4 5]
  Yoink(nil, nil, 2).(A) == Yeet(nil, 3).(A)

  @test Swizzle(+).(rand(3,3)) isa Float64
  @test copy(Swizzle(+)(rand(3,3))) isa Array{Float64, 0}

end
