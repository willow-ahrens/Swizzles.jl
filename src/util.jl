struct Pass end

const pass = Pass()

Base.isequal(::Pass, ::Pass) = true
Base.isequal(::Pass, ::Int) = false
Base.isequal(::Int, ::Pass) = false
Base.isless(::Pass, ::Int) = true
Base.isless(::Int, ::Pass) = false
Base.isless(::Pass, ::Pass) = false

getindexinto(a, b, i::Int) = b[i]
getindexinto(a, b, i::Pass) = a

setindexinto(a, b, i::Int) = map((j, x) -> j == i ? b : x, enumerate(a))
setindexinto(a, b, i::Pass) = a

getindexinto(a, b, i::Tuple{}) = ()
function getindexinto(a, b, i::Union{Tuple{Vararg{<:Union{Int, Pass}}}, AbstractVector{<:Union{Int, Pass}}})
    @boundscheck @assert length(b) == length(i)
    map((a, j)->getindexinto(a, b, j), a, i)
end

setindexinto(a, b, i::Union{NTuple{<:Any, Pass}, AbstractVector{Pass}}) = @boundscheck length(b) == length(i)
function setindexinto(a, b, i::Tuple{Int})
    @boundscheck @assert length(b) == 1
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Pass})
    @boundscheck @assert length(b) == 2
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Pass, Int})
    @boundscheck @assert length(b) == 2
    setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int})
    @boundscheck @assert length(b) == 2
    map((j, x) -> j == i[2] ? b[2] : (j == i[1] ? b[1] : x), enumerate(a))
end
function setindexinto(a, b, i::Union{Tuple{Vararg{<:Union{Int, Pass}}}, AbstractVector{<:Union{Int, Pass}}})
    @boundscheck @assert length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    map((j, x) -> haskey(state, j) ? state[j] : x, enumerate(a))
end
