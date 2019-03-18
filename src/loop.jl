Base.@propagate_inbounds function loop(f::F, itr) where {F}
    for i in itr
        f(i)
    end
    return nothing
end
