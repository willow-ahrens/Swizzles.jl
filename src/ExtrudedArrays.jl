module ExtrudedArrays

using StaticArrays
using Base: RefValue
using Base.Broadcast: Broadcasted, Extruded
using Base.Broadcast: newindexer
using Swizzles.WrapperArrays
using Swizzles.ShallowArrays
using Swizzles.ArrayifiedArrays
using Swizzles.Properties
using Swizzles: combinetuple
using Swizzles: maptuple

using Base: tail

export ExtrudedArray
export keeps, kept
export Keep, Extrude
export keep, extrude
export lift_keeps, stabilize_extrudes_broadcasts

struct Keep end
struct Extrude end
const keep = Keep()
const extrude = Extrude()

Base.:|(::Keep, ::Keep) = Keep()
Base.:|(::Keep, ::Extrude) = Keep()
Base.:|(::Keep, ::Bool) = Keep()

Base.:|(::Extrude, ::Keep) = Keep()
Base.:|(::Extrude, ::Extrude) = Extrude()
Base.:|(::Extrude, k::Bool) = k

Base.:|(::Bool, ::Keep) = Keep()
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
    return ExtrudedArray(arg, maptuple(k-> kept(k) ? Keep() : Extrude(), keeps(arg)...))
end
function ExtrudedArray(arg::ExtrudedArray)
    return ExtrudedArray(arg.arg, maptuple(k-> kept(k) ? Keep() : Extrude(), keeps(arg)...))
end
function ExtrudedArray(arg, keeps)
    return ExtrudedArray{eltype(arg), ndims(arg), typeof(arg), typeof(keeps)}(arg, keeps)
end
function ExtrudedArray(arg::ExtrudedArray, keeps)
    return ExtrudedArray(arg.arg, keeps)
end

Base.parent(arr::ExtrudedArray) = arr.arg
WrapperArrays.adopt(arg::AbstractArray, arr::ExtrudedArray) = ExtrudedArray{eltype(arg), ndims(arr), typeof(arg), typeof(arr.keeps)}(arg, arr.keeps)

#keeps returns a tuple where each element of the tuple specifies whether the
#corresponding dimension is intended to have size 1, possibly using the traits
#keep and extrude (instances of the singleton types `keep` and `extrude`).
keeps(x) = ntuple(ndims(x)) do n
    return size(x, n) != 1
end
keeps(::Tuple{}) = (Keep(),)
keeps(::Tuple{Any}) = (Extrude(),)
keeps(::Tuple) = (Keep(),)
keeps(arr::ExtrudedArray) = arr.keeps
keeps(ext::Extruded) = ext.keeps
keeps(bc::Broadcasted) = combinetuple(|, map(keeps, bc.args)...)



import Base.Broadcast

Base.Broadcast.extrude(x::Extruded) = x
Base.@propagate_inbounds Base.Broadcast.newindex(arg::ExtrudedArray, I::CartesianIndex) = CartesianIndex(Base.Broadcast._newindex(I.I, keeps(arg), maptuple(first, axes(arg)...)))
Base.@propagate_inbounds Base.Broadcast.newindex(arg::ExtrudedArray, I::Integer) = CartesianIndex(Base.Broadcast._newindex((I,), keeps(arg), maptuple(first, axes(arg)...)))
@inline Base.Broadcast.newindexer(A::ExtrudedArray) = (keeps(A), maptuple(first, axes(A)...))
@inline Base.Broadcast.newindex(i::Integer, ::Tuple{Keep}, idefault) = i
@inline Base.Broadcast.newindex(i::Integer, ::Tuple{Extrude}, idefault) = idefault[i]
@inline Base.Broadcast._newindex(I, keep::Tuple{Keep, Vararg{Any}}, Idefault) =
    (I[1], Base.Broadcast._newindex(tail(I), tail(keep), tail(Idefault))...)
@inline Base.Broadcast._newindex(I, keep::Tuple{Extrude, Vararg{Any}}, Idefault) =
    (Idefault[1], Base.Broadcast._newindex(tail(I), tail(keep), tail(Idefault))...)



#=
Properties.return_type(typeof(keeps), T::Type) = Tuple{Vararg{Union{Bool, Extrude, Keep}}}
Properties.return_type(typeof(keeps), T::AbstractArray{N}) where {N} = Tuple{Vararg{N, Union{Bool, Extrude, Keep}}}
function Properties.return_type(typeof(keeps), ::Type{<:StaticArray{S}}) where S <: Tuple
    results = map(S.parameters) do s
        if s isa Integer
            if s == 1
                return Extrude()
            else
                return Keep()
            end
        else
            return Bool
        end
    end
end
Properties.return_type(::typeof(keeps), ::Type{<:ExtrudedArray{<:Any, <:Any, <:Any, _keeps}}) where {_keeps} = _keeps
Properties.return_type(::typeof(keeps), ::Type{<:Tuple{Vararg{Any, N}}) where {N} = N == 1 ? Tuple{Extrude} : Tuple{Keep}
Properties.return_type(::typeof(keeps), ::Type{<:Tuple}) = Tuple{Union{Extrude, Keep}}
Properties.return_type(::typeof(keeps), ::Type{<:Number}) = Tuple{}
Properties.return_type(::typeof(keeps), ::Type{<:ArrayifiedArray{<:Any, <:Any, Arg}}) where {Arg} = return_type(keeps, Arg)
function Properties.return_type(::typeof(keeps), ::Type{<:Broadcasted{<:Any, <:Any, <:Any, Args}}) where {Args<:Tuple}
    Ts = map(arg -> return_type(keeps, arg))
    combinetuple((x, y) -> return_type(|, x, y), map(inferkeeps, Args.parameters)...)
end

Properties.return_type(::typeof(|), ::Type{Keep}, ::Type{Keep}) = Keep
Properties.return_type(::typeof(|), ::Type{Keep}, ::Type{Extrude}) = Keep
Properties.return_type(::typeof(|), ::Type{Keep}, ::Type{Bool}) = Keep
Properties.return_type(::typeof(|), ::Type{Extrude}, ::Type{Keep}) = Keep
Properties.return_type(::typeof(|), ::Type{Bool}, ::Type{Keep}) = Keep

Properties.return_type(::typeof(|), ::Type{Extrude}, ::Type{Extrude}) = Extrude
Properties.return_type(::typeof(|), ::Type{Extrude}, ::Type{Bool}) = Bool
Properties.return_type(::typeof(|), ::Type{Bool}, ::Type{Extrude}) = Bool

Properties.return_type(::typeof(|), ::Type{Bool}, ::Type{Bool}) = Bool

function Properties.return_type(::typeof(|), ::Type{T}, ::Type{S}) where {T<:Union{Keep, Extrude, Bool}, S<:Union{Keep, Extrude, Bool}}
    T = filter(t <: T, [Keep, Extrude, Bool])
    S = filter(s <: S, [Keep, Extrude, Bool])
    return Union{[return_type(|, t, s) for (t, s) in product(T, S)]...}
end
=#


stable_extrude(x) = Extruded(x, maptuple(k-> kept(k) ? Keep() : Extrude(), keeps(x)...), maptuple(first, axes(x)...))

#Add stable Extrudes to all internal broadcast expressions so that the broadcast_getindex does not need dynamic checks.
stabilize_extrudes(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, maptuple(stabilize_extrudes, bc.args...), bc.axes)
stabilize_extrudes(ext::Extruded) = stable_extrude(stabilize_extrudes_broadcasts(ext.x)) #FIXME avoid recomputation in redundant passes
stabilize_extrudes(x) = stable_extrude(stabilize_extrudes_broadcasts(x))
stabilize_extrudes_broadcasts(bc::Broadcasted) = stabilize_extrudes(bc)
stabilize_extrudes_broadcasts(x) = x
function stabilize_extrudes_broadcasts(arr::AbstractArray)
    if iswrapper(arr)
        adopt(stabilize_extrudes_broadcasts(parent(arr)), arr)
    else
        arr
    end
end
function stabilize_extrudes_broadcasts(arr::ArrayifiedArray{T, N}) where {T, N}
    arg = stabilize_extrudes_broadcasts(arr.arg)
    return ArrayifiedArray{T, N, typeof(arg)}(arg)
end

#Add ExtrudedArrays to a broadcast/lazyarray expression so that the keeps of every recognizeable node can be determined from the types.
lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, maptuple(lift_keeps, bc.args...), bc.axes)
lift_keeps(ext::Extruded) = stable_extrude(lift_keeps(ext.x))
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
function lift_keeps(arr::AbstractArray)
    if iswrapper(arr)
        adopt(lift_keeps(parent(arr)), arr)
    elseif ndims(arr) == 0
        return arr
    else
        ExtrudedArray(arr)
    end
end

end
