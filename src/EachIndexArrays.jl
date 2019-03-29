module EachindexArrays

using Swizzles
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.WrapperArrays

using Swizzles: assign!, increment!

struct EachindexArray{T, N, Arg, Inds} <: ShallowArray{T, N, Arg}
    arg::Arg
    inds::Inds
end

Base.parent(arr::EachindexArray) = arr.arg
WrapperArrays.iswrapper(arr::EachindexArray) = true
WrapperArrays.adopt(arg, arr::EachindexArray) = EachindexArray(arg, arr.inds)

function Base.eachindex(arr::EachindexArray)
    return arr.inds
end

struct CartesianTiledIndices{N, Inds <: CartesianIndices{N}, tile} <: ShallowArray{CartesianIndex{N}, N, Inds}
    inds::Inds
    function CartesianTiledIndices{N, Inds, tile}(inds) where {N, Inds, tile}
        @assert tile isa Tuple{Vararg{Int, N}}
        return new{N, Inds, tile}(axes)
    end
end

function CartesianTiledIndices(inds::CartesianIndices{N}, ::Val{tile}) where {N, tile}
    return CartesianTiledIndices{N, typeof(inds), tile}(inds)
end

@generated function Swizzles.assign!(dst, index, src, drive::CartesianTiledIndices{N, Axes, tile}) where {N, Axes, tile}
    thunk = quote
        Base.@_propagate_inbounds_meta
        drive_size = size(drive.inds)
        @nexprs $N n -> II_n = 0:fld(drive_size[n],tile[n])
        @nexprs $N n -> i_n = 1
    end
    for nn = N:-1:0
        thunk = quote
            $thunk
            @nloops $nn ii n -> II_n n -> i_n = 1 + ii_n * tile[n] begin
                axes = @ntuple $N n -> begin
                    if n <= $nn
                        drive.axes[i_n:(i_n + tile[n])]
                    else
                        drive.axes[i_n:end]
                    end
                end
                assign!(dst, index, src, CartesianIndices(axes))
            end
        end
    end
    return thunk
end

struct Tile{_tile_size} <: Swizzles.Intercept end

@inline Tile(_tile_size...) = Tile{_tile_size}()

@inline function parse_tile_size(arr, _tile_size::Tuple{Vararg{Int}})
    return ntuple(n -> n <= length(_tile_size) ? _tile_size[n] : 1, ndims(arr))
end
@inline (ctr::Tile)(arg) = ctr(arrayify(arg))
@inline function(ctr::Tile{_tile_size})(arg::Arg) where {_tile_size, Arg <: AbstractArray}
    if @generated
        tile_size = parse_tile_size(arg, _tile_size)
        return :(return EachindexArray(arg, CartesianTiledIndices(CartesianIndices(arg), tile_size)))
    else
        tile_size = parse_tile_size(arg, _tile_size)
        return EachindexArray(arg, CartesianTiledIndices(CartesianIndices(arg), tile_size))
    end
end

end
