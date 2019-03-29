module EachIndexArrays

struct EachIndexArray{T, N, Arg, Inds} <: ShallowArray{T, N, Arg}
    arg::Arg
    inds::Inds
end

Base.parent(arr::EachIndexArray) = arr.arg
WrapperArrays.iswrapper(arr) = true
WrapperArrays.adopt(arg, arr::EachIndexArray) = EachIndexArray(arg, arr.inds)

function Base.eachindex(arr::EachIndexArray)
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

end
