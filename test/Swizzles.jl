@testset "Swizzles" begin

  A = [1 2 3; 4 5 6; 7 8 9]

  @test swizzle(A, max, (1, drop)) == [3; 6; 9]
  @test swizzle(A, min, (1, drop)) == [1; 4; 7]
  @test swizzle(A, max, (1,)) == [3; 6; 9]
  @test swizzle(A, min, (1,)) == [1; 4; 7]
  @test swizzle(A, max, (drop, 1)) == [7; 8; 9]
  @test swizzle(A, min, (drop, 1)) == [1; 2; 3]
  @test swizzle(A, max, (2, drop)) == [3  6  9]
  @test swizzle(A, min, (2, drop)) == [1  4  7]
  @test swizzle(A, max, (drop, 2)) == [7  8  9]
  @test swizzle(A, min, (drop, 2)) == [1  2  3]

  R = [0; 0; 0;]
  @test swizzle!(R, A, max, (1, drop)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, min, (1, drop)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, max, (1,)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, min, (1,)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, max, (drop, 1)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, min, (drop, 1)) == [1; 2; 3]

  R = [0 0 0]
  @test swizzle!(R, A, max, (2, drop)) == [3  6  9]
  R = [0 0 0]
  @test swizzle!(R, A, min, (2, drop)) == [1  4  7]
  R = [0 0 0]
  @test swizzle!(R, A, max, (drop, 2)) == [7  8  9]
  R = [0 0 0]
  @test swizzle!(R, A, min, (drop, 2)) == [1  2  3]

  @test swizzle(A, max, (1, drop, 2)) == [3; 6; 9]
  @test swizzle(A, min, (1, drop, 2)) == [1; 4; 7]
  @test swizzle(A, max, (drop, 1, 2)) == [7; 8; 9]
  @test swizzle(A, min, (drop, 1, 2)) == [1; 2; 3]

  R = [0; 0; 0;]
  @test swizzle!(R, A, max, (1, drop, 2)) == [3; 6; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, min, (1, drop, 2)) == [1; 4; 7]
  R = [0; 0; 0;]
  @test swizzle!(R, A, max, (drop, 1, 2)) == [7; 8; 9]
  R = [0; 0; 0;]
  @test swizzle!(R, A, min, (drop, 1, 2)) == [1; 2; 3]

  R = [0; 0; 0;]
  @test_throws ArgumentError swizzle!(R, A, nooperator, (drop, 1, 2)) #FIXME call beam
  @test_throws ArgumentError swizzle(A, nooperator, (1, drop, 2))

  @test swizzle(A, nooperator, (2, 1)) == transpose(A)

  R = [0 0 0;
       0 0 0;
       0 0 0]
  @test swizzle!(R, A, min, (2, 1)) == transpose(A)

  @test swizzle([11; 12; 13], nooperator, (2,)) == [11 12 13]
  @test swizzle([11; 12; 13], max, ()) == 13
  @test swizzle((11, 12, 13), max, ()) == 13

  R = [0 0 0]
  @test swizzle!(R, [11; 12; 13], nooperator, (2,)) == [11 12 13]
  R = [0]
  @test swizzle((11, 12, 13), max, ()) == 13

  @test Swizzle(+, ()).(A) == 45
  @test Swizzle(+, (2,)).(A) == [6 15 24]
  @test Swizzle(+, (1,)).(A) == [6; 15; 24]
  @test Swizzle(+, (1,2)).(A) == A
  @test Swizzle(+, (1,2,3)).(A) == A
  @test Swizzle(+, (2,1)).(A) == transpose(A)
  @test Swizzle(+, (2,1,3)).(A) == transpose(A)

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

  @test Swizzle(+, ()).(A.+A) == 90
  @test Swizzle(+, (2,)).(A.+A) == [12 30 48]
  @test Swizzle(+, (1,)).(A.+A) == [12; 30; 48]
  @test Swizzle(+, (1,2)).(A.+A) == A.+A
  @test Swizzle(+, (1,2,3)).(A.+A) == A.+A
  @test Swizzle(+, (2,1)).(A.+A) == transpose(A.+A)
  @test Swizzle(+, (2,1,3)).(A.+A) == transpose(A.+A)

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

  @test SwizzleTo(+, ()).(A) == 45
  @test SwizzleTo(+, (1,)).(A) == [6; 15; 24]
  @test SwizzleTo(+, (2,)).(A) == [12; 15; 18]
  @test SwizzleTo(+, (1,2)).(A) == A
  @test SwizzleTo(+, (1,2,3)).(A) == A
  @test SwizzleTo(+, (2,1)).(A) == transpose(A)
  @test SwizzleTo(+, (2,1,3)).(A) == transpose(A)

  @test_throws ArgumentError BeamTo().(A.+A)
  @test_throws ArgumentError BeamTo(2).(A.+A)
  @test_throws ArgumentError BeamTo(1).(A.+A)
  @test BeamTo(1,2).(A.+A) == A.+A
  @test BeamTo(1,2,3).(A.+A) == A.+A
  @test BeamTo(2,1).(A.+A) == transpose(A.+A)
  @test BeamTo(2,1,3).(A.+A) == transpose(A.+A)

  @test SwizzleTo(+).(A) == 45
  @test SwizzleTo(+, 1).(A) == [6; 15; 24]
  @test SwizzleTo(+, 2).(A) == [12; 15; 18]
  @test SwizzleTo(+, 1, 2).(A) == A
  @test SwizzleTo(+, 2, 1).(A) == transpose(A)
  @test SwizzleTo(+, 2, 1, 3).(A) == transpose(A)

  @test SwizzleTo(+, ()).(A.+A) == 90
  @test SwizzleTo(+, (1,)).(A.+A) == [12; 30; 48]
  @test SwizzleTo(+, (2,)).(A.+A) == [24; 30; 36]
  @test SwizzleTo(+, (1,2)).(A.+A) == A.+A
  @test SwizzleTo(+, (1,2,3)).(A.+A) == A.+A
  @test SwizzleTo(+, (2,1)).(A.+A) == transpose(A.+A)
  @test SwizzleTo(+, (2,1,3)).(A.+A) == transpose(A.+A)

  @test SwizzleTo(+).(A.+A) == 90
  @test SwizzleTo(+, 1).(A.+A) == [12; 30; 48]
  @test SwizzleTo(+, 2).(A.+A) == [24; 30; 36]
  @test SwizzleTo(+, 1, 2).(A.+A) == A.+A
  @test SwizzleTo(+, 2, 1).(A.+A) == transpose(A.+A)
  @test SwizzleTo(+, 2, 1, 3).(A.+A) == transpose(A.+A)

  @test Beam(1).((1, 2)) isa Tuple
  @test Swizzle(+, (1,)).((1, 2)) isa Tuple
  @test Sum(1).((1.0, 2)) isa Float64

  A = rand(1,1)
  B = rand(1,1)

  @test Sum(2).(A.*Beam(2,3).(B)) isa Matrix
  @test Sum().(A.*Beam(2,3).(B)) isa Float64
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
