struct Sequence{T, C, R} <: AbstractVector{T}
  cache::C
  rest::R
end

function getindex(a::Sequence, i::Int)
    @boundscheck i >= 0 || throw
    if i <= length(a.cache)
        return a.cache[i]
    else
        return a.rest[i - length(a.cache)]
    end
end

getindex(a, i::Sequence{Int}) = Sequence(a[i.cache], a[i.rest])



struct Drop end

getindexinto(a, b, ::Drop) = b

setindexinto(a, b, ::Drop) = b

#A = B[I]

function getindexinto(a::Sequence{<:T}, b::Sequence{<:T}, i::Sequence{<:Union{Int, Drop}}) 
    

#drop should return  
axes(sz) = getindexinto(Sequence((inds), repeated(don't call)), Sequence(axes(sz.data), repeated(1)), sz.dims)

inds(sz.data) = getindexinto(axes(sz.data), inds, sz.idims)

#A[I] = B

function setindexinto(a::Sequence{<:T}, b::Sequence{<:T}, i::Sequence{<:Union{Int, Drop}}) 
  

