struct Drop end

const drop = Drop()

Base.isequal(::Drop, ::Drop) = true
Base.isequal(::Drop, ::Int) = false
Base.isequal(::Int, ::Drop) = false
Base.isless(::Drop, ::Int) = true
Base.isless(::Int, ::Drop) = false
Base.isless(::Drop, ::Drop) = false

getindexinto(a, b, i::Int) = b[i]
getindexinto(a, b, i::Drop) = a

setindexinto(a, b, i::Int) = ntuple(j -> j == i ? b : a[j], length(a))
setindexinto(a, b, i::Drop) = a

getindexinto(a, b, i::Tuple{}) = ()
function getindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Drop}}})
    @boundscheck length(a) == length(i)
    r = ntuple(j ->getindexinto(a[j], b, i[j]), length(i))
    return r
end

function getindexinto(a, b, i::AbstractVector{<:Union{Int, Drop}}) #FIXME wrong output
    @boundscheck length(a) == length(i)
    map((x, j)->getindexinto(x, b, j), a, i)
end

function setindexinto(a, b, i::Tuple{Vararg{Drop}})
    @boundscheck length(b) == length(i)
    return a
end
function setindexinto(a, b, i::Tuple{Int})
    @boundscheck length(b) == 1
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Drop})
    @boundscheck length(b) == 2
    setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Drop, Int})
    @boundscheck length(b) == 2
    setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int})
    @boundscheck length(b) == 2
    ntuple(j -> j == i[2] ? b[2] : (j == i[1] ? b[1] : a[j]), length(a))
end

function setindexinto(a, b, i::Tuple{Vararg{<:Union{Int, Drop}}})
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    ntuple(j -> haskey(state, j) ? state[j] : a[j], length(a))
end

function setindexinto(a, b, i::AbstractVector{<:Union{Int, Drop}}) #FIXME wrong output
    @boundscheck length(b) == length(i)
    state = Dict(j => x for (j, x) in zip(i, b))
    map((x, j) -> haskey(state, j) ? state[j] : x, a, 1:length(a))
end
