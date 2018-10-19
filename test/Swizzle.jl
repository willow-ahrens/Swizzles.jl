@testset "Swizzle" begin

  A = [1 2 3; 4 5 6; 7 8 9]

  @test swizzle(A, (1, drop), max) == [3; 6; 9]
  @test swizzle(A, (1, drop), min) == [1; 4; 7]
  @test swizzle(A, (1), max) == [3; 6; 9]
  @test swizzle(A, (1), min) == [1; 4; 7]
  @test swizzle(A, (drop, 1), max) == [7; 8; 9]
  @test swizzle(A, (drop, 1), min) == [1; 2; 3]
  @test swizzle(A, (2, drop), max) == [3  6  9]
  @test swizzle(A, (2, drop), min) == [1  4  7]
  @test swizzle(A, (drop, 2), max) == [7  8  9]
  @test swizzle(A, (drop, 2), min) == [1  2  3]

  R = [0; 0; 0;]
  @test swizzle!(R, A, (1, drop), max) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (1, drop), min) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (1), max) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (1), min) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (drop, 1), max) == [7; 8; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (drop, 1), min) == [1; 2; 3]

  R = [0 0 0]
  @test swizzle!(R, A, (2, drop), max) == [3  6  9]
  R = [0 0 0]
  @test swizzle!(R, A, (2, drop), min) == [1  4  7]
  R = [0 0 0]
  @test swizzle!(R, A, (drop, 2), max) == [7  8  9]
  R = [0 0 0]
  @test swizzle!(R, A, (drop, 2), min) == [1  2  3]

  @test swizzle(A, (1, drop, 2), max) == [3; 6; 9]
  @test swizzle(A, (1, drop, 2), min) == [1; 4; 7]
  @test swizzle(A, (drop, 1, 2), max) == [7; 8; 9]
  @test swizzle(A, (drop, 1, 2), min) == [1; 2; 3]

  R = [0; 0; 0;]
  @test swizzle!(R, A, (1, drop, 2), max) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (1, drop, 2), min) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (drop, 1, 2), max) == [7; 8; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, (drop, 1, 2), min) == [1; 2; 3]

  R = [0; 0; 0;]
  @test_throws ArgumentError swizzle!(R, A, (drop, 1, 2))
  @test_throws ArgumentError swizzle(A, (1, drop, 2))

  @test swizzle(A, (2, 1)) == transpose(A)

  R = [0 0 0;
       0 0 0;
       0 0 0]
  @test swizzle!(R, A, (2, 1), min) == transpose(A)

  @test swizzle([11; 12; 13], (2,)) == [11 12 13]
  @test swizzle([11; 12; 13], (), max) == 13
  @test swizzle((11, 12, 13), (), max) == 13

  R = [0 0 0]
  @test swizzle!(R, [11; 12; 13], (2,)) == [11 12 13]
  R = [0]
  @test swizzle!(R, [11; 12; 13], (), max) == [13]
  @test swizzle((11, 12, 13), (), max) == 13

  @test Swizzler((), +).(A) == 45
  @test Swizzler((2,), +).(A) == [6 15 24]
  @test Swizzler((1,), +).(A) == [6; 15; 24]
  @test Swizzler((1,2), +).(A) == A
  @test Swizzler((1,2,3), +).(A) == A
  @test Swizzler((2,1), +).(A) == transpose(A)
  @test Swizzler((2,1,3), +).(A) == transpose(A)

  @test_throws ArgumentError Beam().(A)
  @test_throws ArgumentError Beam(2).(A)
  @test_throws ArgumentError Beam(1).(A)
  @test Beam(1,2).(A) == A
  @test Beam(1,2,3).(A) == A
  @test Beam(2,1).(A) == transpose(A)
  @test Beam(2,1,3).(A) == transpose(A)

  @test Reduce(+).(A) == 45
  @test Reduce(+, 1).(A) == [12; 15; 18]
  @test Reduce(+, 2).(A) == [6; 15; 24]
  @test Reduce(+, 1, 2).(A) == 45
  @test Reduce(+, 2, 1).(A) == 45
  @test Reduce(+, 2, 1, 3).(A) == 45

  @test Sum().(A) == 45
  @test Sum(1).(A) == [12; 15; 18]
  @test Sum(2).(A) == [6; 15; 24]
  @test Sum(1, 2).(A) == 45
  @test Sum(2, 1).(A) == 45
  @test Sum(2, 1, 3).(A) == 45

  @test Max().(A) == 9
  @test Max(1).(A) == [7; 8; 9]
  @test Max(2).(A) == [3; 6; 9]
  @test Max(1, 2).(A) == 9
  @test Max(2, 1).(A) == 9
  @test Max(2, 1, 3).(A) == 9

  @test Min().(A) == 1
  @test Min(1).(A) == [1; 2; 3]
  @test Min(2).(A) == [1; 4; 7]
  @test Min(1, 2).(A) == 1
  @test Min(2, 1).(A) == 1
  @test Min(2, 1, 3).(A) == 1

  @test Swizzler((), +).(A.+A) == 90
  @test Swizzler((2,), +).(A.+A) == [12 30 48]
  @test Swizzler((1,), +).(A.+A) == [12; 30; 48]
  @test Swizzler((1,2), +).(A.+A) == A.+A
  @test Swizzler((1,2,3), +).(A.+A) == A.+A
  @test Swizzler((2,1), +).(A.+A) == transpose(A.+A)
  @test Swizzler((2,1,3), +).(A.+A) == transpose(A.+A)

  @test_throws ArgumentError Beam().(A.+A)
  @test_throws ArgumentError Beam(2).(A.+A)
  @test_throws ArgumentError Beam(1).(A.+A)
  @test Beam(1,2).(A.+A) == A.+A
  @test Beam(1,2,3).(A.+A) == A.+A
  @test Beam(2,1).(A.+A) == transpose(A.+A)
  @test Beam(2,1,3).(A.+A) == transpose(A.+A)

  @test Reduce(+).(A.+A) == 90
  @test Reduce(+, 1).(A.+A) == [24; 30; 36]
  @test Reduce(+, 2).(A.+A) == [12; 30; 48]
  @test Reduce(+, 1, 2).(A.+A) == 90
  @test Reduce(+, 2, 1).(A.+A) == 90
  @test Reduce(+, 2, 1, 3).(A.+A) == 90

  @test Sum().(A.+A) == 90
  @test Sum(1).(A.+A) == [24; 30; 36]
  @test Sum(2).(A.+A) == [12; 30; 48]
  @test Sum(1, 2).(A.+A) == 90
  @test Sum(2, 1).(A.+A) == 90
  @test Sum(2, 1, 3).(A.+A) == 90

  @test Max().(A.+A) == 18
  @test Max(1).(A.+A) == [14; 16; 18]
  @test Max(2).(A.+A) == [6; 12; 18]
  @test Max(1, 2).(A.+A) == 18
  @test Max(2, 1).(A.+A) == 18
  @test Max(2, 1, 3).(A.+A) == 18

  @test Min().(A.+A) == 2
  @test Min(1).(A.+A) == [2; 4; 6]
  @test Min(2).(A.+A) == [2; 8; 14]
  @test Min(1, 2).(A.+A) == 2
  @test Min(2, 1).(A.+A) == 2
  @test Min(2, 1, 3).(A.+A) == 2

  @test SwizzleTo((), +).(A) == 45
  @test SwizzleTo((1,), +).(A) == [6; 15; 24]
  @test SwizzleTo((2,), +).(A) == [12; 15; 18]
  @test SwizzleTo((1,2), +).(A) == A
  @test SwizzleTo((1,2,3), +).(A) == A
  @test SwizzleTo((2,1), +).(A) == transpose(A)
  @test SwizzleTo((2,1,3), +).(A) == transpose(A)

  @test_throws ArgumentError BeamTo().(A.+A)
  @test_throws ArgumentError BeamTo(2).(A.+A)
  @test_throws ArgumentError BeamTo(1).(A.+A)
  @test BeamTo(1,2).(A.+A) == A.+A
  @test BeamTo(1,2,3).(A.+A) == A.+A
  @test BeamTo(2,1).(A.+A) == transpose(A.+A)
  @test BeamTo(2,1,3).(A.+A) == transpose(A.+A)

  @test ReduceTo(+).(A) == 45
  @test ReduceTo(+, 1).(A) == [6; 15; 24]
  @test ReduceTo(+, 2).(A) == [12; 15; 18]
  @test ReduceTo(+, 1, 2).(A) == A
  @test ReduceTo(+, 2, 1).(A) == transpose(A)
  @test ReduceTo(+, 2, 1, 3).(A) == transpose(A)

  @test SumTo().(A) == 45
  @test SumTo(1).(A) == [6; 15; 24]
  @test SumTo(2).(A) == [12; 15; 18]
  @test SumTo(1, 2).(A) == A
  @test SumTo(2, 1).(A) == transpose(A)
  @test SumTo(2, 1, 3).(A) == transpose(A)

  @test MaxTo().(A) == 9
  @test MaxTo(1).(A) == [3; 6; 9]
  @test MaxTo(2).(A) == [7; 8; 9]
  @test MaxTo(1, 2).(A) == A
  @test MaxTo(2, 1).(A) == transpose(A)
  @test MaxTo(2, 1, 3).(A) == transpose(A)

  @test MinTo().(A) == 1
  @test MinTo(1).(A) == [1; 4; 7]
  @test MinTo(2).(A) == [1; 2; 3]
  @test MinTo(1, 2).(A) == A
  @test MinTo(2, 1).(A) == transpose(A)
  @test MinTo(2, 1, 3).(A) == transpose(A)

  @test SwizzleTo((), +).(A.+A) == 90
  @test SwizzleTo((1,), +).(A.+A) == [12; 30; 48]
  @test SwizzleTo((2,), +).(A.+A) == [24; 30; 36]
  @test SwizzleTo((1,2), +).(A.+A) == A.+A
  @test SwizzleTo((1,2,3), +).(A.+A) == A.+A
  @test SwizzleTo((2,1), +).(A.+A) == transpose(A.+A)
  @test SwizzleTo((2,1,3), +).(A.+A) == transpose(A.+A)

  @test ReduceTo(+).(A.+A) == 90
  @test ReduceTo(+, 1).(A.+A) == [12; 30; 48]
  @test ReduceTo(+, 2).(A.+A) == [24; 30; 36]
  @test ReduceTo(+, 1, 2).(A.+A) == A.+A
  @test ReduceTo(+, 2, 1).(A.+A) == transpose(A.+A)
  @test ReduceTo(+, 2, 1, 3).(A.+A) == transpose(A.+A)

  @test SumTo().(A.+A) == 90
  @test SumTo(1).(A.+A) == [12; 30; 48]
  @test SumTo(2).(A.+A) == [24; 30; 36]
  @test SumTo(1, 2).(A.+A) == A.+A
  @test SumTo(2, 1).(A.+A) == transpose(A.+A)
  @test SumTo(2, 1, 3).(A.+A) == transpose(A.+A)

  @test MaxTo().(A.+A) == 18
  @test MaxTo(1).(A.+A) == [6; 12; 18]
  @test MaxTo(2).(A.+A) == [14; 16; 18]
  @test MaxTo(1, 2).(A.+A) == A.+A
  @test MaxTo(2, 1).(A.+A) == transpose(A.+A)
  @test MaxTo(2, 1, 3).(A.+A) == transpose(A.+A)

  @test MinTo().(A.+A) == 2
  @test MinTo(1).(A.+A) == [2; 8; 14]
  @test MinTo(2).(A.+A) == [2; 4; 6]
  @test MinTo(1, 2).(A.+A) == A.+A
  @test MinTo(2, 1).(A.+A) == transpose(A.+A)
  @test MinTo(2, 1, 3).(A.+A) == transpose(A.+A)

  @test Beam(1).((1, 2)) isa Tuple
  @test Swizzler(1, +).((1, 2)) isa Tuple
  @test Sum(1).((1.0, 2)) isa Float64

  A = rand(1,1)
  B = rand(1,1)

  @test SumTo(1, 3).(A.*Beam(2,3).(B)) isa Matrix
  @test SumTo().(A.*Beam(2,3).(B)) isa Float64
  @test Sum(2).(A.*Beam(2,3).(B)) == A * B

  A = rand(5,7)
  B = rand(7,6)

  @test Sum(2).(A.*Beam(2,3).(B)) ≈ A * B

  @test reshape(Beam(2, 4).(A).*Beam(1, 3).(B), size(A, 1) * size(B, 1), :) == kron(A, B)

  x = rand(3)
  y = rand(4)

  @test reshape(Beam(2).(x).*y, :) == kron(x, y)

  z = Array{Int64,0}(undef)
  z[] = 13

  @test Beam(3).(z) == 13

end
