module ExtrudedArrays

using StaticArrays
using Base: RefValue
using Base.Broadcast: Broadcasted, Extruded
using Base.Broadcast: newindexer
using Swizzles.WrapperArrays
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.Virtuals
using Swizzles.Properties
using Swizzles: combinetuple
using Swizzles: maptuple
using LinearAlgebra

using Base: tail

export ExtrudedArray
export keeps, kept
export Keep, Extrude
export keep, extrude
export lift_keeps, freeze_extrudes

struct Keep end
struct Extrude end
const keep = Keep()
const extrude = Extrude()

Base.:|(::Keep, ::Keep) = keep
Base.:|(::Keep, ::Extrude) = keep
Base.:|(::Keep, ::Bool) = keep

Base.:|(::Extrude, ::Keep) = keep
Base.:|(::Extrude, ::Extrude) = extrude
Base.:|(::Extrude, k::Bool) = k

Base.:|(::Bool, ::Keep) = keep
Base.:|(k::Bool, ::Extrude) = k

kept(::Keep) = true
kept(::Extrude) = false
kept(k::Bool) = k

struct ExtrudedArray{T, N, Arg<:AbstractArray{T, N}, Keeps<:Tuple{Vararg{Any, N}}} <: ShallowArray{T, N, Arg}
    arg::Arg
    keeps::Keeps
    function ExtrudedArray{T, N, Arg, Keeps}(arg::Arg, keeps::Keeps) where {T, N, Arg, Keeps}
        @assert keeps isa Tuple{Vararg{Any, N}}
        @assert all(ntuple(n -> kept(keeps[n]) || size(arg, n) == 1, N)) "$(size(arg)) $(keeps)"
        return new{T, N, Arg, Keeps}(arg, keeps)
    end
end

function ExtrudedArray(arg)
    return ExtrudedArray(arg, maptuple(k-> kept(k) ? keep : extrude, keeps(arg)...))
end
function ExtrudedArray(arg::ExtrudedArray)
    return ExtrudedArray(arg.arg, maptuple(k-> kept(k) ? keep : extrude, keeps(arg)...))
end
function ExtrudedArray(arg, keeps)
    return ExtrudedArray{eltype(arg), ndims(arg), typeof(arg), typeof(keeps)}(arg, keeps)
end
function ExtrudedArray(arg::ExtrudedArray, keeps)
    return ExtrudedArray(arg.arg, keeps)
end

@inline Base.parent(arr::ExtrudedArray) = arr.arg
@inline WrapperArrays.adopt(arg::AbstractArray, arr::ExtrudedArray) = ExtrudedArray{eltype(arg), ndims(arr), typeof(arg), typeof(arr.keeps)}(arg, arr.keeps)

#keeps returns a tuple where each element of the tuple specifies whether the
#corresponding dimension is intended to have size 1, possibly using the traits
#keep and extrude (instances of the singleton types `keep` and `extrude`).
keeps(x) = ntuple(ndims(x)) do n
    return size(x, n) != 1
end
keeps(::Tuple{}) = (keep,)
keeps(::Tuple{Any}) = (extrude,)
keeps(::Tuple) = (keep,)
function keeps(arr::Adjoint)
    arg_keeps = keeps(parent(arr))
    if length(arg_keeps) == 0
        return (extrude, extrude)
    elseif length(arg_keeps) == 1
        return (extrude, arg_keeps[1])
    elseif length(arg_keeps) == 2
        return (arg_keeps[2], arg_keeps[1])
    else
        throw(ArgumentError("TODO"))
    end
end
function keeps(arr::Transpose)
    arg_keeps = keeps(parent(arr))
    if length(arg_keeps) == 0
        return (extrude, extrude)
    elseif length(arg_keeps) == 1
        return (extrude, arg_keeps[1])
    elseif length(arg_keeps) == 2
        return (arg_keeps[2], arg_keeps[1])
    else
        throw(ArgumentError("TODO"))
    end
end
keeps(arr::ExtrudedArray) = arr.keeps
keeps(ext::Extruded) = ext.keeps
keeps(bc::Broadcasted) = combinetuple(|, map(keeps, bc.args)...)
keeps(arr::ArrayifiedArray) = keeps(arr.arg)
keeps(arr::StaticArray) = ntuple(ndims(arr)) do n
    return size(arr, n) == 1 ? keep : extrude
end



import Base.Broadcast

@inline Base.Broadcast.extrude(x::Extruded) = x
Base.@propagate_inbounds Base.Broadcast.newindex(arg::ExtrudedArray, I::CartesianIndex) = CartesianIndex(Base.Broadcast._newindex(I.I, keeps(arg), maptuple(first, axes(arg)...)))
Base.@propagate_inbounds Base.Broadcast.newindex(arg::ExtrudedArray, I::Integer) = CartesianIndex(Base.Broadcast._newindex((I,), keeps(arg), maptuple(first, axes(arg)...)))
@inline Base.Broadcast.newindexer(A::ExtrudedArray) = (keeps(A), maptuple(first, axes(A)...))
@inline Base.Broadcast.newindex(i::Integer, ::Tuple{Keep}, idefault) = i
@inline Base.Broadcast.newindex(i::Integer, ::Tuple{Extrude}, idefault) = idefault[i]
@inline Base.Broadcast._newindex(I, keeps::Tuple{Keep, Vararg{Any}}, Idefault) =
    (I[1], Base.Broadcast._newindex(tail(I), tail(keeps), tail(Idefault))...)
@inline Base.Broadcast._newindex(I, keeps::Tuple{Extrude, Vararg{Any}}, Idefault) =
    (Idefault[1], Base.Broadcast._newindex(tail(I), tail(keeps), tail(Idefault))...)



static_extrude(x) = Extruded(x, maptuple(k-> kept(k) ? keep : extrude, keeps(x)...), maptuple(first, axes(x)...))

#Add stable Extrudes to all internal broadcast expressions so that the broadcast_getindex does not need dynamic checks.
_freeze_extrudes(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, maptuple(_freeze_extrudes, bc.args...), bc.axes)
_freeze_extrudes(ext::Extruded) = static_extrude(freeze_extrudes(ext.x)) #FIXME avoid recomputation in redundant passes
_freeze_extrudes(x) = static_extrude(freeze_extrudes(x))
freeze_extrudes(bc::Broadcasted) = _freeze_extrudes(bc)
freeze_extrudes(x) = x
function freeze_extrudes(arr::AbstractArray)
    if iswrapper(arr)
        adopt(freeze_extrudes(parent(arr)), arr)
    else
        arr
    end
end
function freeze_extrudes(arr::ArrayifiedArray)
    return arrayify(freeze_extrudes(arr.arg))
end

#Add ExtrudedArrays to a broadcast/lazyarray expression so that the keeps of every recognizeable node can be determined from the types.
lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, maptuple(lift_keeps, bc.args...), bc.axes)
lift_keeps(ext::Extruded) = static_extrude(lift_keeps(ext.x))
lift_keeps(arr::ExtrudedArray) = ExtrudedArray(lift_keeps(ext.x))
lift_keeps(arr::ArrayifiedArray) = arrayify(lift_keeps(arr.arg))
function lift_keeps(x)
    if ndims(x) == 0
        return x
    else
        return ExtrudedArray(arrayify(x))
    end
end
lift_keeps(tu::Tuple) = tu
function lift_keeps(arr::Adjoint{T}) where {T}
    arg = lift_keeps(parent(arr))
    return Adjoint{T, typeof(arg)}(arg)
end
function lift_keeps(arr::Transpose{T}) where {T}
    arg = lift_keeps(parent(arr))
    return Transpose{T, typeof(arg)}(arg)
end
function lift_keeps(arr::AbstractArray)
    if ndims(arr) == 0
        return arr
    else
        ExtrudedArray(arr)
    end
end

function Virtuals.virtual(root, ::Type{Keep})
    return keep
end
function Virtuals.virtual(root, ::Type{Extrude})
    return extrude
end
function Virtuals.virtualize(root, ::Type{ExtrudedArray{T, N, Arg, Keeps}}) where {T, N, Arg, Keeps}
    arg = virtualize(:($root.arg), Arg)
    keeps = virtualize(:($root.keeps), Keeps)
    return ExtrudedArray{T, N, typeof(arg), typeof(keeps)}(arg, keeps)
end

Base.:|(::Keep, ::Virtual{Bool}) = keep
Base.:|(::Virtual{Bool}, ::Keep) = keep
Base.:|(::Extrude, y::Virtual{Bool}) = y
Base.:|(x::Virtual{Bool}, ::Extrude) = x
Base.:|(x::Virtual{Bool}, ::Virtual{Bool}) = Virtual{Bool}(:($(x.ex) | $(y.ex)))

keeps(::Virtual{Tuple{}}) = (keep,)
keeps(::Virtual{Tuple{Any}}) = (extrude,)
keeps(::Virtual{<:Tuple}) = (keep,)


end
