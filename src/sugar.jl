using MacroTools
using Swizzles

macro sugar(ex)
    op = nothing
    reg = nothing
    vars = nothing
    init = nothing
    elem = nothing

        if @capture(ex, [vars__] = elem_)
    elseif @capture(ex, [vars__] += elem_) op = :+
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
    else
        return :(error("TODO"))
    end

    function dim(var, env)
        if var == :(:)
            return env[gensym()] = length(env) + 1
        elseif var isa Symbol
            if !haskey(dims, var)
                env[var] = length(env) + 1
            end
            return universe[var]
        elseif var isa Integer
            return nil
        end
        error("TODO")
    end

    N = length(vars)
    Dict(vars[n] => n for n in 1:N)

    function beamify(ex, env)
        if @capture(ex, arr_[inds__])
            return :($(Beam(map(ind -> dim(ind, env), inds)...)).($arr))
        elseif ex isa Expr
	    return Expr(ex.head, map(arg->beamify(arg, env), ex.args)...)
        else
            return ex
        end
    end

    init_env = Dict(vars[n] => n for n in 1:N)
    if init === nothing
        init = ()
    else
        init = (Base.Broadcast.__dot__(beamify(init, init_env)), )
    end

    elem_env = Dict()
    mask = Val((map(var-> dim(var, elem_env), vars)...,))
    elem = Base.Broadcast.__dot__(beamify(elem, elem_env))

    if reg === nothing
        return esc(:(Swizzle($op, $mask).($(init...), $elem)))
    else
        return esc(:($reg .= Swizzle($op, $mask).($(init...), $elem)))
    end
end
