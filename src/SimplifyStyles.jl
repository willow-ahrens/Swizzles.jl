module SimplifyStyles

using Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

using LinearAlgebra
using MacroTools
using Rewrite
using Rewrite: Rule, PatternRule, Property

using Swizzles
using Swizzles.Properties
using Swizzles.Antennae
using Swizzles.ValArrays
using Swizzles.Virtuals
using Swizzles: SwizzledArray


export Simplify



"""
    rewriteable(root, T::Type)

Given a root expression of type T, produce the most detailed constructor
expression you can to create a new object that should be equal to root.

# Examples
```jldoctest
julia> A = [1 2 3 4; 5 6 7 8]'
4×2 LinearAlgebra.Adjoint{Int64,Array{Int64,2}}:
 1  5
 2  6
 3  7
 4  8
julia> r = rewriteable(:A, typeof(A))
:(LinearAlgebra.Adjoint(parent(A)::Array{Int64,2}))
julia> eval(Main, r) == A
true
```
"""
function rewriteable(root, T::Type)
    syms = Dict{Symbolic, Any}()
    return (Term(rewriteable(root, T, syms)), syms)
end

function rewriteable(root, T::Type, syms)
    s = Symbolic(gensym())
    syms[s] = root
    return :($s::$T)
end

function rewriteable(root, ::Type{<:ValArray{<:Any, val}}, syms) where {val}
    ValArray(val)
end

function rewriteable(root, ::Type{<:Adjoint{<:Any, Arg}}, syms) where Arg
    arg = rewriteable(:($root.parent), Arg, syms)
    :(Adjoint($arg))
end

function rewriteable(root, ::Type{<:Transpose{<:Any, Arg}}, syms) where Arg
    arg = rewriteable(:($root.parent), Arg, syms)
    :(Transpose($arg))
end

function rewriteable(root, ::Type{<:Broadcasted{<:Any, <:Any, F, Args}}, syms) where {F, Args<:Tuple}
    args = map(((i, arg),) -> rewriteable(:($root.args[$i]), arg, syms), enumerate(Args.parameters))
    f = instance(F)
    if f !== nothing
        :($(Antenna(something(f)))($(args...)))
    else
        return :($root::T)
    end
end

function rewriteable(root, T::Type{<:SwizzledArray{<:Any, <:Any, Op, mask, Init, Arg}}, syms) where {Op, mask, Init, Arg}
    init = rewriteable(:($root.init), Init, syms)
    arg = rewriteable(:($root.arg), Arg, syms)
    op = instance(Op)
    if op !== nothing
        :($(Swizzle(something(op), mask))($init, $arg))
    else
        return :($root::T)
    end
end



"""
Transforms a Term (used for Rewrite.jl) into an evaluable julia expression
"""
function evaluable(term::Term, syms)
    return evaluable(term.ex, syms)
end

function evaluable(ex::Expr, syms)
    return Expr(ex.head, map(arg->evaluable(arg, syms), ex.args)...)
end

function evaluable(ex::Symbolic, syms)
    return syms[ex]
end

function evaluable(ex, syms)
    return ex
end



"""
Virtually evaluate a Term (used for Rewrite.jl)
"""
function veval(term::Term, syms)
    return veval(evaluable(term))
end

function veval(ex::Expr)
    if @capture(ex, _ex::_T)
        @assert T isa Type
        return virtualize(ex, T)
    elseif @capture(ex, f_(args__))
        #TODO should we check anything here?
        return f(map(veval, args)...)
    else
        ArgumentError("Cannot virtually evaluate expression $ex")
    end
end

"""
Stores rules for simplification.
"""
struct SimplificationSpec
    rules::Rules
    context::Context
end


"""
Helper function that converts a vanilla term to one that uses Antennas.
"""
antenna_term(t::Term) = Term(antenna_expr(t.ex))
antenna_expr(v::Variable) = v
function antenna_expr(ex::Expr) :: Expr
    if @capture(ex, f_(args__))
        if f isa Union{Symbolic, Variable} # return f(b_t(arg1), b_t(arg2), b_t(arg3), ...)
            return Expr(:call, f, map(antenna_term, args)...)
        elseif f isa Expr
            throw(ArgumentError("nonbroadcastable term: $ex"))
        elseif f isa Symbol
            throw(ArgumentError("nonbroadcastable term: $ex"))
        else # return Antenna(f)(b_t(arg1), b_t(arg2), b_t(arg3))
            return Expr(:call, Antenna(f), map(antenna_expr, args)...)
        end
    end
    throw(ArgumentError("can't convert to antenna version: $ex"))
end


"""
Helper function that converts a vaniall rule to one that uses Antennas.
"""
function antenna_rule(rule::PatternRule) :: PatternRule
    if !isempty(rule.ps)
        throw(ArgumentError("can't convert rule with properties: $r"))
    end
    return PatternRule(antenna_term(rule.left), antenna_term(rule.right))
end


"""
Returns the default SimplificationSpec.
"""
function default_spec()
    @vars x y z
    @vars swz

    pointwise_rules = Array{Rule, 1}([
        PatternRule(@term(x * (y + z)), @term(x * y + x * z))
    ])
    append!(equalities, map(antenna_rule, pointwise_rules))
    other_rules = Array{Rule, 1}([
        PatternRule(@term(swz(init, arg)), @term((swz::Swizzle)(arg)) if swz isa Swizzle && isnilpotent(swz.op, init) )
    ])
    append!(equalities, other_rules)

    properties = []

    return SimplificationSpec(Rules(equalities), Context(properties))
end

DEFAULT_SPEC = default_spec();


"""
    simplify(arr)

Apply global rules to simplify the array expression `arr`.
Currently only supports broadcast expressions.
"""
@generated function simplify(arr)
    term, syms = rewriteable(:arr, arr)
    simple_term = Rewrite.with_context(DEFAULT_SPEC.context) do
        Rewrite.normalize(term, DEFAULT_SPEC.rules)
    end
    simple_expr = evaluable(simple_term, syms)

    return quote
        $simple_expr
    end
end


"""
    Simplify

Simplify is an abstract type which when passed as the first expression of
broadcasted returns a simplified version of the second Broadcasted expression.

Due to the way the dot syntactic sugar for broadcasts works, this also
allows Simplify to intercept broadcasted expressions. See the example below.

# Example
```jldoctest
julia> A, B = [1 2 3; 4 5 6], [100 200 300]
([1 2 3; 4 5 6], [100 200 300])

julia> Simplify().(A)
2×3 Array{Int64,2}:
 1  2  3
 4  5  6

julia> Simplify().(A .+ B)
2×3 Array{Int64,2}:
 101  202  303
 104  205  306
```
"""
struct Simplify end
#Base.broadcasted(::Simplify, b::Broadcasted) = simplify(lift_names(b))
Base.broadcasted(::Simplify, b::Broadcasted) = simplify(b)
Base.broadcasted(::Simplify, x) = broadcasted(Simplify(), broadcasted(identity, x))


end
