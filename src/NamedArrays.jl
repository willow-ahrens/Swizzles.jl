module NamedArrays
    using StaticArrays
    using Base: RefValue
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer
    using Swizzles.WrapperArrays
    using Swizzles.ShallowArrays
    using Swizzles.ArrayifiedArrays
    using Swizzles.Properties

    export NamedArray, name, lift_names

    struct NamedArray{T, N, Arg<:AbstractArray{T, N}, name} <: ShallowArray{T, N, Arg}
        arg::Arg
        function NamedArray{T, N, Arg, name}(arg::Arg) where {T, N, Arg, name}
            return new{T, N, Arg, name}(arg)
        end
    end

    function NamedArray(arg, name)
        return NamedArray{eltype(arg), ndims(arg), typeof(arg), name}(arg)
    end

    Base.parent(arr::NamedArray) = arr.arg
    function WrapperArrays.adopt(arg::AbstractArray, arr::NamedArray{T, N, Arg, name}) where {T, N, Arg, name}
        arg === arr.arg || throw(ArgumentError("cannot change object identity of named array"))
        NamedArray{T, N, Arg, name}(arg)
    end

    name(arr::NamedArray{<:Any, <:Any, <:Any, _name}) where {_name} = _name
    name(::Type{<:NamedArray{<:Any, <:Any, <:Any, _name}}) where {_name} = _name

    function name(obj, stuff)
        if haskey(stuff, obj)
            return stuff[obj]
        else
            stuff[obj] = Symbol("obj$(length(stuff) + 1)")
        end
    end

    lift_names(obj) = lift_names(obj, Dict())
    function lift_names(obj, stuff)
        arr = arrayify(obj)
        return NamedArray(arr, name(arr, stuff))
    end
    function lift_names(arr::ArrayifiedArray, stuff)
        return arrayify(lift_names(arr.arg, stuff))
    end
    lift_names(ext::Extruded, stuff) = Extruded(lift_names(ext.x, stuff), ext.keeps, ext.defaults)
    lift_names(bc::Broadcasted{Style}, stuff) where {Style} = Broadcasted{Style}(bc.f, map(arg->lift_names(arg, stuff), bc.args), bc.axes)
end
