module ExtrudedArrays
    using StaticArrays
    using Base: RefValue
    using Base.Broadcast: Broadcasted, Extruded
    using Base.Broadcast: newindexer
    using Swizzles.WrapperArrays
    using Swizzles.ArrayifiedArrays
    using Swizzles.ShallowArrays
    using Swizzles: combinetuple

    export ExtrudedArray
    export inferkeeps, keeps, kept, lift_keeps
    export StableKeep, ShakyKeep, Extrude, Dynamic

    struct StableKeep end
    struct ShakyKeep end
    struct Extrude end
    struct Dynamic end

    Base.:|(::StableKeep, ::StableKeep) = StableKeep()
    Base.:|(::StableKeep, ::ShakyKeep) = ShakyKeep()
    Base.:|(::StableKeep, ::Extrude) = StableKeep()
    Base.:|(::StableKeep, ::Dynamic) = ShakyKeep()
    Base.:|(::StableKeep, ::Bool) = ShakyKeep()

    Base.:|(::ShakyKeep, ::StableKeep) = ShakyKeep()
    Base.:|(::ShakyKeep, ::ShakyKeep) = ShakyKeep()
    Base.:|(::ShakyKeep, ::Extrude) = ShakyKeep()
    Base.:|(::ShakyKeep, ::Dynamic) = ShakyKeep()
    Base.:|(::ShakyKeep, ::Bool) = ShakyKeep()

    Base.:|(::Extrude, ::StableKeep) = StableKeep()
    Base.:|(::Extrude, ::ShakyKeep) = ShakyKeep()
    Base.:|(::Extrude, ::Extrude) = Extrude()
    Base.:|(::Extrude, ::Dynamic) = Dynamic()
    Base.:|(::Extrude, k::Bool) = k

    Base.:|(::Dynamic, ::StableKeep) = ShakyKeep()
    Base.:|(::Dynamic, ::ShakyKeep) = ShakyKeep()
    Base.:|(::Dynamic, ::Extrude) = Dynamic()
    Base.:|(::Dynamic, ::Dynamic) = Dynamic()
    Base.:|(::Dynamic, ::Bool) = Dynamic()

    Base.:|(::Bool, ::StableKeep) = ShakyKeep()
    Base.:|(::Bool, ::ShakyKeep) = ShakyKeep()
    Base.:|(k::Bool, ::Extrude) = k
    Base.:|(::Bool, ::Dynamic) = Dynamic()

    kept(::StableKeep) = true
    kept(::ShakyKeep) = true
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
    keeps(::Tuple{}) = (StableKeep(),)
    keeps(::Tuple{Any}) = (Extrude(),)
    keeps(::Tuple) = (StableKeep(),)
    keeps(::ExtrudedArray{<:Any, <:Any, <:Any, _keeps}) where {_keeps} = _keeps
    keeps(ext::Extruded) = ext.keeps
    keeps(bc::Broadcasted) = combinetuple(|, map(keeps, bc.args)...)

    inferkeeps(T::Type) = throw(MethodError(inferkeeps, (T,)))
    function inferkeeps(::Type{<:StaticArray{S}}) where S <: Tuple
        return map(S.parameters) do s
            if s isa Integer
                if s == 1
                    return Extrude()
                else
                    return StableKeep()
                end
            else
                return Dynamic()
            end
        end
    end
    inferkeeps(::Type{<:ExtrudedArray{<:Any, <:Any, <:Any, _keeps}}) where {_keeps} = _keeps
    inferkeeps(::Type{<:Tuple}) = (true,)
    inferkeeps(::Type{<:Tuple{<:Any}}) = (false,)
    inferkeeps(::Type{<:Number}) = ()
    inferkeeps(::Type{<:RefValue}) = ()
    inferkeeps(Arr::Type{<:ArrayifiedArray{<:Any, <:Any, Arg}}) where {Arg} = inferkeeps(Arg)
    inferkeeps(::Type{<:Broadcasted{<:Any, <:Any, <:Any, Args}}) where {Args<:Tuple} = combinetuple(|, map(inferkeeps, Args.parameters)...)

    lift_keeps(x) = ExtrudedArray(x)
    lift_keeps(x::ArrayifiedArray{T, N}) where {T, N} = ArrayifiedArray{T, N}(lift_keeps(x.arg))
    lift_keeps(x::StaticArray) = x
    lift_keeps(x::Tuple) = x
    lift_keeps(x::Number) = x
    lift_keeps(x::RefValue) = x
    lift_keeps(bc::Broadcasted{Style}) where {Style} = Broadcasted{Style}(bc.f, map(lift_keeps, bc.args))
end
