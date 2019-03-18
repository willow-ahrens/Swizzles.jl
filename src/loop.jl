Base.@propagate_inbounds function loop(f::F, itr) where {F}
    for i in itr
        f(i)
    end
    return nothing
end

@generated function loop(f::F, itr::CartesianIndices{N}) where {F, N}
    thunk = :(f(CartesianIndex($([Symbol("i$n") for n in 1:N]...))))
    for n in 1:N
        thunk = quote
            for $(Symbol("i$n")) = itr.indices[$n]
                $thunk
            end
        end
    end
    return quote
        Base.@_propagate_inbounds_meta
        $thunk
        return nothing
    end
end
