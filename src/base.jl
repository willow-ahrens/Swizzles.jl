##
# Base hotfixes
##
module BaseHacks
    @info "All your Base belong to us. Activate BaseHacks!!!"

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
end
