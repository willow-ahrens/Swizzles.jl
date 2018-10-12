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

setindexinto(a, b, i::Int) = ntuple(j -> j == i ? b : a[j], length(a))
setindexinto(a, b, i::Pass) = a

getindexinto(a, b, i::Tuple{}) = ()
function getindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Pass}}})
    @boundscheck length(a) == length(i)
    r = ntuple(j ->getindexinto(a[j], b, i[j]), length(i))
    return r
end

function getindexinto(a, b, i::AbstractVector{<:Union{Int, Pass}}) #FIXME wrong output
    @boundscheck length(a) == length(i)
    map((x, j)->getindexinto(x, b, j), a, i)
end

function setindexinto(a, b, i::Tuple{Vararg{Pass}})
    @boundscheck length(b) == length(i)
    return a
end
function setindexinto(a, b, i::Tuple{Int})
    @boundscheck length(b) == 1
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Pass})
    @boundscheck length(b) == 2
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Pass, Int})
    @boundscheck length(b) == 2
    setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int})
    @boundscheck length(b) == 2
    ntuple(j -> j == i[2] ? b[2] : (j == i[1] ? b[1] : a[j]), length(a))
end

function setindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Pass}}})
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    ntuple(j -> haskey(state, j) ? state[j] : a[j], length(a))
end

function setindexinto(a, b, i::AbstractVector{<:Union{Int, Pass}}) #FIXME wrong output
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    map((x, j) -> haskey(state, j) ? state[j] : x, a, 1:length(a))
end
