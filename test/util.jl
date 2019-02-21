@testset "util" begin
    Is = [(),
          (1,),
          (drop,),
          (1, 2,),
          (drop, 2,),
          (2, drop,),
          (1, 2, 4,),
          (1, 2, drop, 4,),
          (1, drop, 3,)]
    for I in Is
        @test length(masktuple(+, -, I)) == length(I)
        @test masktuple(+, -, I) isa Tuple{Vararg{Int}}
        for j in 1:length(I)
            if I[j] isa Drop
                @test(masktuple(+, -, I)[j] == j)
            else
               @test(masktuple(+, -, I)[j] == -I[j])
            end
        end
        @test imasktuple(+, -, I) isa Tuple{Vararg{Int}}
        @test length(imasktuple(+, -, I)) == max(0, I...)
        for j in 1:length(I)
            if !(I[j] isa Drop)
                @test(imasktuple(+, -, I)[I[j]] == -j)
            end
        end
        for j in 1:max(0, I...)
            if !(j in I)
                @test(imasktuple(+, -, I)[j] == j)
            end
        end
    end
end
