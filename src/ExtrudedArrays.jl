module ExtrudedArrays
    using StaticArrays
    using Base: RefValue
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer
    using Swizzle.WrapperArrays
    using Swizzle.BroadcastedArrays

    export ExtrudedArray
    export keeps, lift_keeps

    struct ExtrudedArray{T, N, Arg<:AbstractArray{T, N}, keeps} <: ShallowArray{T, N, Arg}
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
    WrapperArrays.adopt(arg::AbstractArray, arr::ExtrudedArray) = ExtrudedArray(arg)

    #keeps is a complicated function. It returns a tuple where each element of
    #the tuple specifies whether the corresponding dimension is intended to have
    #size 1. The complicated aspect of keeps is that while it should work on
    #BroadcastedArray, it must also work on the type wrapped by BroadcastedArray.
    #This way, lift_keeps only needs to use ArrayifiedArrays when it's creating
    #an ExtrudedArray.
    keeps(x) = newindexer(x)[1]
    keeps(ext::Extruded) = ext.keeps

    keeps(T::Type) = throw(MethodError(keeps, (T,)))
    keeps(::ExtrudedArray{<:Any, <:Any, <:Any, _keeps}) where {_keeps} = _keeps
    keeps(::Type{<:ExtrudedArray{<:Any, <:Any, <:Any, _keeps}}) where {_keeps} = _keeps
    keeps(Arr::Type{<:StaticArray{<:Any, <:Any, N}}) where {N} = ntuple(n -> length(axes(Arr)[n]) == 1, N)
    keeps(Arr::Type{<:BroadcastedArray{<:Any, <:Any, Arg}}) where {Arg} = keeps(Arg)
    keeps(::Type{<:Tuple}) = (true,)
    keeps(::Type{<:Tuple{<:Any}}) = (false,)
    keeps(::Type{<:Number}) = ()
    keeps(::Type{<:RefValue}) = ()

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
    lift_keeps(x::Number) = x
    lift_keeps(x::RefValue) = x
    lift_keeps(x::BroadcastedArray{T, N}) where {T, N} = BroadcastedArray{T, N}(lift_keeps(x.arg))
    lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(lift_keeps, bc.args))
end
