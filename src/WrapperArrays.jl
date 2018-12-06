module WrapperArrays

using Base.Broadcast: broadcast_axes, BroadcastStyle
using Base: dataids, unaliascopy, unalias

import LinearAlgebra

export iswrapper, adopt, storage

export ShallowArray

#=
    This file defines
        1. What it means to be a "simple wrapper array". Any simple wrapper
        must define the `adopt` function, which describes how to construct
        analogous wrapper arrays with new parents, and specialize Base.parent.
        2. Corresponding methods for "simple wrapper arrays" in Base.
        3. A conveience type `ShallowArray` which defines several pass-through
        methods for easy construction of wrapper arrays.
    Ideally, numbers 1 and 2 should live in Adapt.jl, (note that `adapt` can be
    implemented with adopt).

    Wrappers need to define
        parent
        adopt
        iswrapper?
        dataids?
        unaliascopy?
        alias?

=#

"""
    adopt(arg, child)

Wrap `arg` in an analogous wrapper array to `child`. This function should
create an array with the same semantics as `child`. Generally, if `p` is
mostly the same array as `parent(x)`, it should hold that `adopt(p, x)` is
mostly the same array as `x`.

See also: [`parent`](@ref), [`iswrapper`](@ref)
"""
adopt(arg, arr) = throw(MethodError(adopt, (arg, arr)))

"""
    storage(arr)

Return the deepest ancestor of the array `arr`. If `arr` is a wrapper array,
returns `storage(parent(arr))`. Otherwise, returns `arr`.

See also: [`parent`](@ref), [`iswrapper`](@ref)
"""
function storage(f::F, arr) where {F}
    if iswrapper(arr)
        return parent(arr)
    else
        return arr
    end
end

"""
    iswrapper(arr)

A trait function which returns `true` if and only if `arr !== parent(arr)`. We
reccommend specializing this function with hard-coded implementations for custom
array types to improve type inference of wrapper array functions like
[`storage`](@ref).

See also: [`parent`](@ref)
"""
iswrapper(arr) = arr !== parent(arr)

#Base
iswrapper(::Array) = false

iswrapper(::LinearAlgebra.Transpose) = true
adopt(arg::AbstractVecOrMat, arr::LinearAlgebra.Transpose) = LinearAlgebra.transpose(arg)

iswrapper(::LinearAlgebra.Adjoint) = true
adopt(arg::AbstractVecOrMat, arr::LinearAlgebra.Adjoint) = LinearAlgebra.adjoint(arg)

iswrapper(::SubArray) = true
adopt(arg, arr::SubArray) = SubArray(arg, parentindices(arr))

iswrapper(::LinearAlgebra.LowerTriangular) = true
adopt(arg, arr::LinearAlgebra.LowerTriangular) = LinearAlgebra.LowerTriangular(arg)

iswrapper(::LinearAlgebra.UnitLowerTriangular) = true
adopt(arg, arr::LinearAlgebra.UnitLowerTriangular) = LinearAlgebra.UnitLowerTriangular(arg)

iswrapper(::LinearAlgebra.UpperTriangular) = true
adopt(arg, arr::LinearAlgebra.UpperTriangular) = LinearAlgebra.UpperTriangular(arg)

iswrapper(::LinearAlgebra.UnitUpperTriangular) = true
adopt(arg, arr::LinearAlgebra.UnitUpperTriangular) = LinearAlgebra.UnitUpperTriangular(arg)

iswrapper(::LinearAlgebra.Diagonal) = true
adopt(arg, arr::LinearAlgebra.Diagonal) = LinearAlgebra.Diagonal(arg)

iswrapper(::Base.ReshapedArray) = true
adopt(arg, arr::Base.ReshapedArray) = reshape(arg, arr.dims)

iswrapper(::PermutedDimsArray) = true
function adopt(arg::Arg, arr::PermutedDimsArray{<:Any,N,perm,iperm}) where {T,N,perm,iperm,Arg<:AbstractArray{T, N}}
    return PermutedDimsArray{T,N,perm,iperm,Arg}(arg)
end


end
