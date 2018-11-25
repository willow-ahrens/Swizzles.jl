module ExtrudedArrays
    using StaticArrays
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer
    using Swizzle.WrappedArrays
    using Swizzle.BroadcastedArrays

    export ExtrudedArray
    export keeps, lift_keeps

    struct ExtrudedArray{T, N, Arg<:AbstractArray{T, N}, keeps} <: WrappedArray{T, N, Arg}
        arg::Arg
        function ExtrudedArray{T, N, Arg, keeps}(arg::Arg) where {T, N, keeps, Arg}
            @assert keeps isa Tuple{Vararg{Bool, N}}
            @assert all(ntuple(n -> keeps[n] || length(axes(arg, n)) == 1, N))
            return new{T, N, Arg, keeps}(arg)
        end
    end

    function ExtrudedArray(arg)
        arr = arrayify(arg)
        return ExtrudedArray{eltype(arr), ndims(arr), typeof(arr), keeps(arr)}(arr)
    end

    Base.parent(arr::ExtrudedArray) = arr.arg

    #keeps is a complicated function. It returns a tuple where each element of the tuple specifies whether the corresponding dimension is intended to have size 1. The complicated aspect of keeps is that while it should work on BroadcastedArray, it must also work on the type wrapped by BroadcastedArray. This way, lift_keeps only needs to use BroadcastedArrays when it's creating an ExtrudedArray.
    keeps(x) = newindexer(x)[1]
    keeps(ext::Extruded) = ext.keeps

    keeps(T::Type) = throw(MethodError(keeps, (T,)))
    keeps(::ExtrudedArray{<:Any, <:Any, <:Any, _keeps}) where {_keeps} = _keeps
    keeps(::Type{<:ExtrudedArray{<:Any, <:Any, <:Any, _keeps}}) where {_keeps} = _keeps
    keeps(Arr::Type{<:StaticArray{<:Any, <:Any, N}}) where {N} = ntuple(n -> length(axes(Arr)[n]) == 1, N)
    keeps(Arr::Type{<:BroadcastedArray}) = keeps(parenttype(Arr))
    keeps(::Type{<:Tuple}) = (true,)
    keeps(::Type{<:Tuple{<:Any}}) = (false,)

    function keeps(bc::Broadcasted)
        args = map(keeps, bc.args)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end
    function keeps(::Type{<:Broadcasted{<:Any, <:Any, <:Any, Args}}) where {Args<:Tuple}
        args = map(keeps, Args.parameters)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end

    lift_keeps(x) = ExtrudedArray(x)
    lift_keeps(x::StaticArray) = x
    lift_keeps(x::Tuple) = x
    lift_keeps(x::BroadcastedArray{<:Any, <:Any, <:Tuple}) = x
    lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(lift_keeps, bc.args))
end
