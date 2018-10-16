using BenchmarkTools

include("getbenchmarks.jl")

loadparams!(suite, BenchmarkTools.load("params.json")[1], :evals, :samples);

results = run(suite)

println(IOContext(stdout, :verbose => true), results)
