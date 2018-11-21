module ExtrudedArrays
    using StaticArrays
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer

    export ExtrudedArray
    export keepify, keeps

    struct ExtrudedArray{T, N, Arg<:AbstractArray{T, N}, keeps} <: WrapperArray{T, N, Arg}
        arg::Arg
        function ExtrudedArray{T, N, Arg, keeps}(arg::Arg) where {T, N, keeps, Arg}
            @assert keeps isa Tuple{Vararg{Bool, N}}
            @assert all(ntuple(keeps[n] || length(axes(arg, n)) == 1, N))
            return new{T, N, keeps, Arg}(arg)
        end
    end

    function ExtrudedArray(arg)
        arr = arrayify(arg)
        return ExtrudedArray{eltype(arr), ndims(arr), typeof(arr), keeps(arr)}(arr)
    end

    Base.parent(arr::ExtrudedArray) = arr.arg

    keeps(x) = newindexer(x)[1]
    keeps(ext::Extruded) = ext.keeps
    keeps(::Type) = throw(MethodError())
    keeps(::ExtrudedArray{<:Any, <:Any, <:Any, _keeps}) where {_keeps} = _keeps
    keeps(::Type{ExtrudedArray{<:Any, <:Any, <:Any, _keeps}}) where {_keeps} = _keeps
    function keeps(bc::Broadcasted)
        args = map(keeps, bc.args)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end
    function keeps(::Broadcasted{<:Any, <:Any, <:Any, Args<:Tuple}) where {Args}
        args = map(keeps, Args.parameters)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end

    lift_keeps(x) = ExtrudedArray(x)
    lift_keeps(x::StaticArray) = ExtrudedArray(x)
    lift_keeps(b::Broadcasted) = Broadcasted(b.f, map(lift_keeps, b.args))
end
