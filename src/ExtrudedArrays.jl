using Base.Broadcast: newindexer

struct ExtrudedArray{T, N, keeps, Arg} <: AbstractArray{T, N}
    arg::Arg
    function ExtrudedArray{T, N, keeps, Arg}(arg::Arg) where {T, N, keeps, Arg}
        @assert keeps isa Tuple{Vararg{Bool, N}}
        return new{T, N, keeps, Arg}(arg)
    end
end

function ExtrudedArray(arg)
    keeps = newindexer(arg)[1]
    return ExtrudedArray{eltype(T), length(keeps), keeps, typeof(arg)}(arg)
end

function Base.show(io::IO, ext::ExtrudedArray{T, N}) where {T, N}
    print(io, ExtrudedArray{T, N}) #Showing the arg type (although maybe useful since it's allowed to differ), will likely be redundant.
    print(io, '(', ext.arg, ext.exts, ')')
    nothing
end
