using MacroTools
using Swizzles

macro swizzle(ex)
    op = nothing
    reg = nothing
    vars = nothing
    init = nothing
    elem = nothing

        if @capture(ex, [vars__] += elem_) op = :+
    elseif @capture(ex, [vars__] *= elem_) op = :*
    elseif @capture(ex, [vars__] |= elem_) op = :|
    elseif @capture(ex, [vars__] &= elem_) op = :&
    elseif @capture(ex, [vars__] <op_>= elem_)
    elseif @capture(ex, [vars__] = init_ += elem_) op = :+
    elseif @capture(ex, [vars__] = init_ *= elem_) op = :*
    elseif @capture(ex, [vars__] = init_ |= elem_) op = :|
    elseif @capture(ex, [vars__] = init_ &= elem_) op = :&
    elseif @capture(ex, [vars__] = init_ <op_>= elem_)
    elseif @capture(ex, reg_[vars__] = elem_) init = :($reg[$(vars...)])
    elseif @capture(ex, reg_[vars__] += elem_) init = :($reg[$(vars...)]); op = :+
    elseif @capture(ex, reg_[vars__] *= elem_) init = :($reg[$(vars...)]); op = :*
    elseif @capture(ex, reg_[vars__] |= elem_) init = :($reg[$(vars...)]); op = :|
    elseif @capture(ex, reg_[vars__] &= elem_) init = :($reg[$(vars...)]); op = :&
    elseif @capture(ex, reg_[vars__] <op_>= elem_) init = :($reg[$(vars...)])
    elseif @capture(ex, reg_[vars__] = init_ += elem_) op = :+
    elseif @capture(ex, reg_[vars__] = init_ *= elem_) op = :*
    elseif @capture(ex, reg_[vars__] = init_ |= elem_) op = :|
    elseif @capture(ex, reg_[vars__] = init_ &= elem_) op = :&
    elseif @capture(ex, reg_[vars__] = init_ <op_>= elem_)
    elseif @capture(ex, [vars__] = elem_)
    else
        return :(error("TODO"))
    end

    N = length(vars)
    dims = Dict(vars[n] => n for n in 1:N)

    function dim(var)
        if !haskey(dims, var)
            dims[var] = length(dims) + 1
        end
        return dims[var]
    end

    function beamify(ex)
        if @capture(ex, arr_[vars__])
            return :($(Beam((map(dim, vars)...))).($arr))
        elseif ex isa Expr
	    return Expr(ex.head, map(beamify, ex.args)...)
        else
            return ex
        end
    end

    if init === nothing
        init = ()
    else
        init = (Base.Broadcast.__dot__(beamify(init)),)
    end

    elem = Base.Broadcast.__dot__(beamify(elem))
    mask = Val((1:length(vars)...,))
    if reg === nothing
        return esc(:(Swizzle($op, $mask).($(init...), $elem)))
    else
        return esc(:($reg .= Swizzle($op, $mask).($(init...), $elem)))
    end
end
