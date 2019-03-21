##
# Base hotfixes
##
module BaseHacks
    @info "activating BaseHacks..."

    import Base.IteratorsMD: inc
    import Base: tail

    function unsafe_inc(i::Int64)
        Base.llvmcall("""
        %i = add nsw i64 %0, 1
        ret i64 %i
        """, Int64, Tuple{Int64}, i)
    end

    @inline inc(state::Tuple{Int}, start::Tuple{Int}, stop::Tuple{Int}) = (unsafe_inc(state[1]),)
    @inline function inc(state, start, stop)
         if state[1] < stop[1]
           return (unsafe_inc(state[1]),tail(state)...)
         end
         newtail = inc(tail(state), tail(start), tail(stop))
         (start[1], newtail...)
    end

    @inline function Base.getindex(r::AbstractUnitRange, s::AbstractUnitRange{<:Integer})
        @boundscheck checkbounds(r, s)
        f = first(r)
        st = oftype(f, f + first(s)-1)
        Base._range(st, nothing, nothing, length(s)) #gotta use this version of range instead of const-propping through the kwarg version.
    end

    #FIXME This is a separate eachindex for scalar cases so that swizzles can use
    #a different thunk for the scalar case and avoid recursion depth limiting.
    Base.eachindex(args::AbstractArray{<:Any, 0}...) = (CartesianIndex(),)

    Base.@propagate_inbounds function Base.eachindex(::IndexLinear, A::AbstractArray, B::AbstractArray...)
        indsA = eachindex(IndexLinear(), A)
        @boundscheck Base._all_match_first(X->eachindex(IndexLinear(), X), indsA, B...) || Base.throw_eachindex_mismatch(IndexLinear(), A, B...)
        indsA
    end

    Base.@propagate_inbounds function Base.eachindex(::IndexCartesian, A::AbstractArray, B::AbstractArray...)
        axsA = axes(A)
        @boundscheck Base._all_match_first(axes, axsA, B...) || Base.throw_eachindex_mismatch(IndexCartesian(), A, B...)
        CartesianIndices(axsA)
    end

    #=
    Base.@propagate_inbounds function Base.getindex(iter::CartesianIndices{N,<:NTuple{N,Base.OneTo}}, I::Vararg{Int, N}) where {N}
        @boundscheck checkbounds(iter, I...)
        CartesianIndex(I)
    end

    Base.@propagate_inbounds function Base.getindex(iter::CartesianIndices{N,R}, I::Vararg{Int, N}) where {N,R}
        @boundscheck checkbounds(iter, I...)
        CartesianIndex(I .- first.(Base.axes1.(iter.indices)) .+ first.(iter.indices))
    end
    =#


    @info "BaseHacks is online. all your Base are belong to us."
end
