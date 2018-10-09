struct Skip end

const skip = Skip()

Base.isequal(::Skip, ::Skip) = true
Base.isequal(::Skip, ::Int) = false
Base.isequal(::Int, ::Skip) = false
Base.isless(::Skip, ::Int) = true
Base.isless(::Int, ::Skip) = false
Base.isless(::Skip, ::Skip) = false

getindexinto(a, b, i::Int) = b[i]
getindexinto(a, b, i::Skip) = a

setindexinto(a, b, i::Int) = map((j, x) -> j == i ? b : x, enumerate(a))
setindexinto(a, b, i::Skip) = a

getindexinto(a, b, i::Tuple{}) = ()
getindexinto(a, b, i::Union{NTuple{<:Any, <:Union{Int, Skip}}, AbstractVector{<:Union{Int, Skip}}}) = map((a, j)->getindex(a, b, j), a, j)

setindexinto(a, b, i::Union{NTuple{<:Any, Skip}, AbstractVector{Skip}}) = @boundscheck length(b) == length(i)
function setindexinto(a, b, i::Tuple{Int}) =
  @boundscheck length(b) == 1
  setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Int, Skip}) =
  @boundscheck length(b) == 2
  setindexinto(a, b[1], i[1])
end
function setindexinto(a, b, i::Tuple{Skip, Int}) =
  @boundscheck length(b) == 2
  setindexinto(a, b[2], i[2])
end
function setindexinto(a, b, i::Tuple{Int, Int}) =
  @boundscheck length(b) == 2
  map((j, x) -> j == i[2] ? b[2] : (j == i[1] ? b[1] : x), enumerate(a))
end
function setindexinto(a, b, i::Union{NTuple{<:Any, <:Union{Int, Skip}}, AbstractVector{<:Union{Int, Skip}}}) =
  @boundscheck length(b) == length(i)
  state = Dict(j => x for (j, x) in zip(i, b))
  map((j, x) -> haskey(state, j) ? state[j] : x, enumerate(a))
end
