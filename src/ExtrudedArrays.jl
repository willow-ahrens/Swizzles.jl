module ExtrudedArrays
    using StaticArrays
    using Base: RefValue
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer
    using Swizzles.WrapperArrays
    using Swizzles.ArrayifiedArrays
    using Swizzles.ShallowArrays
    using Swizzles.Properties
    using Swizzles: combinetuple

    export ExtrudedArray
    export keeps, kept, lift_keeps
    export Keep, Extrude

    struct Keep end
    struct Extrude end

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

    struct ExtrudedArray{T, N, Arg<:AbstractArray{T, N}, keeps} <: ShallowArray{T, N, Arg}
        arg::Arg
        function ExtrudedArray{T, N, Arg, keeps}(arg::Arg) where {T, N, keeps, Arg}
            @assert keeps isa Tuple{Vararg{Bool, N}}
            @assert all(ntuple(n -> kept(keeps[n]) || size(arg, n) == 1, N)) "$(size(arg)) $(keeps)"
            return new{T, N, Arg, keeps}(arg)
        end
    end

    function ExtrudedArray(arg)
        arr = arrayify(arg)
        return ExtrudedArray{eltype(arr), ndims(arr), typeof(arr), keeps(arr)}(arr)
    end

    Base.parent(arr::ExtrudedArray) = arr.arg
    WrapperArrays.adopt(arg::AbstractArray, arr::ExtrudedArray) = ExtrudedArray{eltype(arg), ndims(arr), typeof(arg), keeps(arr)}(arg)

    #keeps is a complicated function. It returns a tuple where each element of
    #the tuple specifies whether the corresponding dimension is intended to have
    #size 1. The complicated aspect of keeps is that while it should work on
    #ArrayifiedArray, it must also work on the type wrapped by ArrayifiedArray.
    #This way, lift_keeps only needs to use ArrayifiedArrays when it's creating
    #an ExtrudedArray.
    keeps(x) = ntuple(ndims(x)) do n
        return size(x, n) != 1
    end
    keeps(::Tuple{}) = (Keep(),)
    keeps(::Tuple{Any}) = (Extrude(),)
    keeps(::Tuple) = (Keep(),)
    keeps(::ExtrudedArray{<:Any, <:Any, <:Any, _keeps}) where {_keeps} = _keeps
    keeps(ext::Extruded) = ext.keeps
    keeps(bc::Broadcasted) = combinetuple(|, map(keeps, bc.args)...)

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


    lift_keeps(x) = ExtrudedArray(x)
    lift_keeps(x::ArrayifiedArray{T, N}) where {T, N} = ArrayifiedArray{T, N}(lift_keeps(x.arg))
    lift_keeps(x::StaticArray) = x
    lift_keeps(x::Tuple) = x
    lift_keeps(x::Number) = x
    lift_keeps(x::RefValue) = x
    lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(lift_keeps, bc.args))
end
