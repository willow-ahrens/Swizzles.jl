@testset "util" begin
    refgetindexinto(a, B, i::Union{Integer, Drop}) = refgetindexinto((a,), B, (i,))[1]
    refgetindexinto(A, B, I::Tuple) = (refgetindexinto(A, B, [I...,])...,)
    function refgetindexinto(A, B, I::AbstractVector)
        R = similar(I, Any)
        for j in eachindex(I)
            if I[j] isa Drop
                R[j] = A[j]
            else
                R[j] = B[I[j]]
            end
        end
        return map(identity, R)
    end

    refsetindexinto(A::Tuple, b, i::Union{Integer, Drop}) = refsetindexinto(A, (b,), (i,))
    refsetindexinto(A::AbstractVector, b, i::Union{Integer, Drop}) = refsetindexinto(A, (b,), (i,))
    refsetindexinto(A::Tuple, B, I) = (refsetindexinto([A...,], B, I)...,)
    function refsetindexinto(A::AbstractVector, B, I)
        R = similar(A, Any)
        for j in eachindex(A)
            R[j] = A[j]
        end
        for j in eachindex(I)
            if !(I[j] isa Drop)
                R[I[j]] = B[j]
            end
        end
        return map(identity, R)
    end

    As = [[],
          [-1.0],
          [-1; -2],
          [-1; -2; -3],
          [-1.0; -2.0]]
    Bs = [[],
          [11.0],
          [11; 12],
          [11.0; 12.0],
          [11; 12; 13]]
    Is = [[],
          [1],
          [drop],
          [1; 2],
          [drop; 2],
          [2; drop],
          [2; 2; 1],
          [1; 2; 4],
          [1; drop; 3; 3]]
    As = vcat(As, [(A...,) for A in As], [-1, -1.0])
    Bs = vcat(Bs, [(B...,) for B in Bs], [11, -11.0])
    Is = vcat(Is, [(I...,) for I in Is], [-1, 1, drop])
    for (A, B, I, (ref_f, test_f)) in Iterators.product(As, Bs, Is, ((refsetindexinto, setindexinto), (refgetindexinto, getindexinto)))
        try
            ref_f(A, B, I)
        catch E
            if !isa(@test_throws(typeof(E), test_f(A, B, I)), Test.Pass)
                println("Failed On: $test_f($A, $B, $I)")
                println("Expected Error:", throw(E))
                break
            end
            continue
        end
        try
            if !isa(@test(test_f(A, B, I) == ref_f(A, B, I)), Test.Pass)
                println("Failed On: $test_f($A, $B, $I)")
                break
            end
            if !isa(@test(typeof(test_f(A, B, I)) == typeof(ref_f(A, B, I))), Test.Pass)
                println("Failed On: $test_f($A, $B, $I)")
                break
            end
        catch E
            throw(E)
            break
        end
    end
end
