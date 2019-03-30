
struct RecursiveIndices{T, N, Inds <: AbstractArray{T, N}} <: AbstractArray{T, N}
    indices::Inds
    limits::NTuple{Int, N}
end

@generated loop(f::F, inds::RecursiveIndices) where {F}
    if n > N
        return :(loop(f, CartesianIndices($(axes...))))
    else
        a = gensym()
        l = gensym()
        r = gensym()
        s = gensym()
        return quote
            if length(ind_axes[$n]) > 10
                $a = itr.indices[$n][1:10]
                $(rloop(n + 1, N, cat(axes, a)))
            else
                $s = fld1(length(ax), 2)
                $l = itr.indices[$n][1:$s]
                $r = itr.indices[$n][$s + 1:end]
                $(rloop(n + 1, N, cat(axes, l)))
                $(rloop(n + 1, N, cat(axes, r)))
            end
        end
    end
end


_RecursiveIndices_loop(n, N, axes)
    if n > N
        return :(loop(f, CartesianIndices($(axes...))))
    else
        a = gensym()
        l = gensym()
        r = gensym()
        s = gensym()
        return quote
            if length(ind_axes[$n]) > 10
                $a = itr.indices[$n][1:10]
                $(rloop(n + 1, N, cat(axes, a)))
            else
                $s = fld1(length(ax), 2)
                $l = itr.indices[$n][1:$s]
                $r = itr.indices[$n][$s + 1:end]
                $(rloop(n + 1, N, cat(axes, l)))
                $(rloop(n + 1, N, cat(axes, r)))
            end
        end
    end
end
