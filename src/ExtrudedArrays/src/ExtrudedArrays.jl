module ExtrudedArrays

    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer

    export ExtrudedArray
    export keepify, kept

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
        keeps = ntuple(keeps[n] || length(axes(arg, n)) == 1, N)
        return ExtrudedArray{eltype(arr), ndims(arr), typeof(arr), keeps}(arr)
    end

    Base.parent(arr::ExtrudedArray) = arr.arg

    keepify(x) = ExtrudedArray(x)
    keepify(b::Broadcasted) = Broadcasted(b.f, map(keepify, b.args))

    kept(x) = newindexer(x)[1]
    kept(ext::Extruded) = ext.keeps
    kept(::Type) = throw(MethodError())
    kept(::ExtrudedArray{<:Any, <:Any, <:Any, keeps}) where {keeps} = keeps
    kept(::Type{ExtrudedArray{<:Any, <:Any, <:Any, keeps}}) where {keeps} = keeps
    function kept(bc::Broadcasted)
        args = map(kept, bc.args)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end
    function kept(::Broadcasted{<:Any, <:Any, <:Any, Args<:Tuple}) where {Args}
        args = map(kept, Args.parameters)
        N = maximum(map(length, args))
        return ntuple(n -> any(arg -> length(arg) >= n && arg[n], args), N)
    end
end
