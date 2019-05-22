module EachindexArrays

using Swizzles
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.WrapperArrays

using Base.Cartesian

using Swizzles: assign!, increment!

export EachindexArray, CartesianTiledIndices, Tile

struct EachindexArray{T, N, Arg, Indices} <: ShallowArray{T, N, Arg}
    arg::Arg
    indices::Indices
end

@inline EachindexArray(arg::AbstractArray, indices) = EachindexArray{eltype(arg), ndims(arg), typeof(arg), typeof(indices)}(arg, indices)

@inline Base.parent(arr::EachindexArray) = arr.arg
@inline WrapperArrays.iswrapper(arr::EachindexArray) = true
@inline WrapperArrays.adopt(arr::EachindexArray, arg) = EachindexArray(arg, arr.indices)

@inline function Base.eachindex(arr::EachindexArray)
    return arr.indices
end

@inline function Base.eachindex(arr::EachindexArray, args::AbstractArray...)
    return arr.indices
end

struct CartesianTiledIndices{N, Indices <: CartesianIndices{N}, Chunks <: Tuple{Vararg{Any, N}}} <: ShallowArray{CartesianIndex{N}, N, Indices}
    indices::Indices
    chunks::Chunks
    function CartesianTiledIndices{N, Indices, Chunks}(indices, chunks) where {N, Indices, Chunks}
        return new{N, Indices, Chunks}(indices, chunks)
    end
end

@inline Base.parent(arr::CartesianTiledIndices) = arr.indices
@inline WrapperArrays.iswrapper(arr::CartesianTiledIndices) = true

@inline function CartesianTiledIndices(indices::Indices, chunks::Chunks) where {Indices, Chunks}
    CartesianTiledIndices{ndims(indices), Indices, Chunks}(indices, chunks)
end



@generated function Swizzles.assign!(dst, index, src, drive::CartesianTiledIndices{N, Indices, chunks}) where {N, Indices, chunks}
    return quote
        Base.@_propagate_inbounds_meta
        @nloops $N i n -> 1:drive.chunks[n]:length(drive.indices.indices[n]) begin
            tile = @ntuple $N n -> drive.indices.indices[n][i_n:min(i_n + drive.chunks[n] - 1, length(drive.indices.indices[n]))]
            assign!(dst, index, src, CartesianIndices(tile))
        end
    end
end

@generated function Swizzles.increment!(op::Op, dst, index, src, drive::CartesianTiledIndices{N, Indices, chunks}) where {Op, N, Indices, chunks}
    return quote
        Base.@_propagate_inbounds_meta
        @nloops $N i n -> 1:drive.chunks[n]:length(drive.indices.indices[n]) begin
            tile = @ntuple $N n -> drive.indices.indices[n][i_n:min(i_n + drive.chunks[n] - 1, length(drive.indices.indices[n]))]
            increment!(op, dst, index, src, CartesianIndices(tile))
        end
    end
end

struct Tile{_Chunks} <: Swizzles.Intercept
    _chunks::_Chunks
end

@inline Tile(_chunks...) = Tile{typeof(_chunks)}(_chunks)

@inline (ctr::Tile)(arg) = ctr(arrayify(arg))
@inline function(ctr::Tile{<:Tuple{Vararg{Any, _N}}})(arg::Arg) where {_N, Arg <: AbstractArray}
    if @generated
        chunks = ntuple(n -> n <= _N ? :(ctr._chunks[$n]) : 1, ndims(arg))
        return :(return EachindexArray(arg, CartesianTiledIndices(CartesianIndices(arg), ($(chunks...),))))
    else
        chunks = ntuple(n -> n <= _N ? ctr._chunks[n] : 1, ndims(arg))
        return EachindexArray(arg, CartesianTiledIndices(CartesianIndices(arg), chunks))
    end
end

end
