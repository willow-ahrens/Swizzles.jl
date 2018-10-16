using BenchmarkTools

include("getbenchmarks.jl")

tune!(suite);

BenchmarkTools.save("params.json", params(suite));
