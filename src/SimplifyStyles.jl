module SimplifyStyles

using Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

using LinearAlgebra
using MacroTools
using Rewrite
using Rewrite: Rule, PatternRule, Property

using Swizzles
using Swizzles.Antennae
using Swizzles.ValArrays
using Swizzles: SwizzledArray


export Simplify


SymbolExpr = Union{Symbol, Expr}  # Type alias for convenience


"""
    reprexpr(root::Union{Symbol, Expr}, T::Type) :: Expr

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
julia> r = reprexpr(:A, typeof(A))
:(LinearAlgebra.Adjoint(parent(A)::Array{Int64,2}))
julia> eval(Main, r) == A
true
```
"""
reprexpr(root::SymbolExpr, T) :: Expr = :($root::$T)

function reprexpr(root::SymbolExpr, ::Type{<:ValArray{<:Any, val}}) where {val}
    val_arr = ValArray(val)
    :($val_arr)
end

function reprexpr(root::SymbolExpr,
                  ::Type{<:Adjoint{<:Any, Arg}}) :: Expr where Arg
    arg_expr = reprexpr(:($root.parent), Arg)
    :($Adjoint($arg_expr))
end

function reprexpr(root::SymbolExpr,
                  ::Type{<:Transpose{<:Any, Arg}}) :: Expr where Arg
    arg_expr = reprexpr(:($root.parent), Arg)
    :($Transpose($arg_expr))
end

function reprexpr(root::SymbolExpr,
                  ::Type{<:Broadcasted{<:Any, <:Any, F, Args}}) :: Expr where {F, Args<:Tuple}
    antenna = Antenna(F.instance)
    arg_exprs = tuple_reprexpr(:($root.args), Args)

    :($antenna($(arg_exprs...)))
end

function reprexpr(root::SymbolExpr,
                  ::Type{<:SwizzledArray{<:Any, <:Any, Op, Mask, Init, Arg}}) :: Expr where {Op, Mask, Init, Arg}
    swizzle = Swizzle(Op.instance, Mask)
    init_expr = reprexpr(:($root.init), Init)
    arg_expr = reprexpr(:($root.arg), Arg)

    :($swizzle($init_expr, $arg_expr))
end

function tuple_reprexpr(root::SymbolExpr,
                        ::Type{TType}) :: Array{Expr, 1} where TType<:Tuple
    [reprexpr(:($root[$idx]), EType)
         for (idx, EType) in enumerate(TType.parameters)]
end


"""
Transforms a SymbolExpr into a Term (used for Rewrite.jl)
"""
function expr_to_term(ex::SymbolExpr) :: Tuple{Term, Dict{Symbolic, SymbolExpr}}
    sym_to_ex = Dict{Symbolic, SymbolExpr}()
    ex = addSymbolics(ex, sym_to_ex)
    (Term(ex), sym_to_ex)
end;

function addSymbolics(ex::Symbol,
                      sym_to_ex::Dict{Symbolic, SymbolExpr}) :: Symbolic
    s = Symbolic(gensym())
    sym_to_ex[s] = ex
    return s
end

function addSymbolics(ex::Expr,
                      sym_to_ex::Dict{Symbolic, SymbolExpr}) :: Union{Expr, Symbolic}
    if @capture(ex, f_(args__))
        return :($f(
            $(map(arg->addSymbolics(arg, sym_to_ex), args)...)
        ))
    elseif @capture(ex, arg_::T_) && !(T isa Union{Symbol, Expr})
        s = Symbolic(gensym())
        sym_to_ex[s] = arg
        return s
    else
        s = Symbolic(gensym())
        sym_to_ex[s] = ex
        return s
    end
end


"""
Transforms a Term (used for Rewrite.jl) into an SymbolExpr.
"""
function term_to_expr(term::Term,
                      sym_to_ex::Dict{Symbolic, SymbolExpr}) :: SymbolExpr
    return removeSymbolics(term.ex, sym_to_ex)
end

function removeSymbolics(ex::Expr,
                         sym_to_ex::Dict{Symbolic, SymbolExpr}) :: SymbolExpr
    if @capture(ex, f_(args__))
        return :($f(
            $(map(arg->removeSymbolics(arg, sym_to_ex), args)...)
        ))
    end
    throw(ArgumentError("non expr transformable: $ex"))
end

function removeSymbolics(s::Symbolic,
                         sym_to_ex::Dict{Symbolic, SymbolExpr}) :: SymbolExpr
    return sym_to_ex[s]
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

    equalities = Array{Rule, 1}([
        PatternRule(@term(x * (y + z)), @term(x * y + x * z))
    ])
    append!(equalities, map(antenna_rule, equalities))

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
    expr = reprexpr(:arr, arr)
    term, sym_to_ex = expr_to_term(expr)
    simple_term = Rewrite.with_context(DEFAULT_SPEC.context) do
        Rewrite.normalize(term, DEFAULT_SPEC.rules)
    end
    simple_expr = term_to_expr(simple_term, sym_to_ex)

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