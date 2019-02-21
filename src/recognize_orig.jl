using SpecialSets
using Rewrite
using Rewrite: PatternRule, EvalRule, Rule, Rules, Associative, Commutative, Property
using MacroTools

broadcast_term(t::Term) = Term(broadcast_term(get(t)))
broadcast_term(t) = t
function broadcast_term(ex::Expr)
    if ex.head == :call
        f, args = ex.args[1], ex.args[2:end]
        if f isa Union{Symbolic, Variable}
            return Expr(:call, f, map(broadcast_term, args)...)
        elseif f isa Expr
            throw(ArgumentError("nonbroadcastable term: $ex"))
        elseif f isa Symbol
            throw(ArgumentError("nonbroadcastable term: $ex"))
        else
            return Expr(:call, Broadcaster(f), map(broadcast_term, args)...)
        end
    end
    throw(ArgumentError("nonbroadcastable term: $ex"))
end

broadcast_rule(r) = throw(ArgumentError("nonbroadcastable rule: $r"))
function broadcast_rule(r::PatternRule)
    if length(r.ps) != 0
        throw(ArgumentError("nonbroadcastable rule: $r"))
    end
    return PatternRule(broadcast_term(r.left), broadcast_term(r.right))
end

function broadcast_rule(r::EvalRule)
    r.name isa Function || throw(ArgumentError("nonbroadcastable rule: $r"))
    r.f isa Function || throw(ArgumentError("nonbroadcastable rule: $r"))
    EvalRule(Broadcaster(r.name), Broadcaster(r.f))
end

struct Distributive <: Property
    f
    g
end

broadcast_property(p) = throw(ArgumentError("nonbroadcastable rule: $p"))
broadcast_property(a::Associative) = Associative(Broadcaster(a.f))
broadcast_property(c::Commutative) = Commutative(Broadcaster(c.f))
broadcast_property(d::Distributive) = Distributive(Broadcaster(d.f), Broadcaster(d.g))

struct ArbRule <: Rule
    f
end
Rewrite.normalize(t::Term, r::ArbRule) = r.f(t)

#f(g(x, y), g(x, z)) = g(x, f(y, z))


struct Library
    equalities::Rules
    matches::Rules
    context::Context
end

function default_library()
    @vars x y z

    equalities = [
        PatternRule(@term(x * y + x * z), @term(x * (y + z))),
        PatternRule(@term(y / x + z / x), @term((y + z) / x)),
        PatternRule(@term(x - y), @term(x + -y)),
        PatternRule(@term(x + -x), @term(zero(x))),
        PatternRule(@term(-(x + y)), @term(-y + -x)),
        PatternRule(@term(-(-x)), @term(x)),
        PatternRule(@term(-1 * x), @term(-x)),
        PatternRule(@term(-x * y), @term(-(x * y))),
        PatternRule(@term(x^0), @term(one(x))),
        PatternRule(@term(x^0.0), @term(one(x))),
        PatternRule(@term(x^1), @term(x)),
        PatternRule(@term(x^1.0), @term(x)),
        PatternRule(@term(abs(-x)), @term(abs(x))),
        PatternRule(@term(abs(x * y)), @term(abs(x) * abs(y))),
        PatternRule(@term(abs(inv(x))), @term(inv(abs(x)))),
        EvalRule(+),
        EvalRule(-),
        EvalRule(*),
        EvalRule(abs),
        EvalRule(&),
        EvalRule(|),
        EvalRule(!)
    ]

    append!(equalities, map(broadcast_rule, equalities))

    append!(equalities, [
        ArbRule(t -> begin
                @vars s b x α
                for σ in match(@term(s(b(α, x))), t)
                    σ[s] isa Swizzler || continue
                    σ[b] isa Broadcaster || continue
                    isvalid(Distributive(σ[s].op, σ[b].f)) || continue
                    all(map(in(axes(mask(σ[s]))), keepdims(σ[x]))) || continue
                    all(map(isa(Int), mask(σ[s])[keepdims(σ[x])])) || continue
                    return replace(@term(b($(Swizzler(mask(σ[s]), nooperator))(α), s(x))), σ)
                end
                return t
            end)
    ])

    properties = [
        Distributive(+, *),
        Associative(+),
        Commutative(+),
        Associative(*),
        Commutative(*),
        Associative(&),
        Commutative(&),
        Associative(|),
        Commutative(|)
    ]

    append!(properties, map(broadcast_property, properties))

    return Library(Rules(equalities), Rules(), Context(properties))
end

LIBRARY = default_library()

function recognize(item::Term, lib::Library=LIBRARY, default=item)
    with_context(()->begin
        norm_item = normalize(item, lib.equality)
        match_item = normalize(item, lib.matches)
        if norm_item == match_item
            return default
        else
            match_item
        end
    end, lib.context)
end

function termtransform(ex, ts)
    s = gensym()
    ts[s] = ex
    return Symbolic(s)
end
function termtransform(ex::Expr, ts)
    if @capture(ex, arg_::T_) && !(T isa Union{Symbol, Expr})
        s = gensym()
        if T isa QuoteNode
            T = T.value
        end
        ts[s] = arg
        #return Symbolic(s, TypeSet(T_))
        return Symbolic(s)
    elseif @capture(ex, f_(args__))
        return :($f($(map(arg->termtransform(arg, ts), args)...)))
    else
        s = gensym()
        ts[s] = ex
        return Symbolic(s)
    end
end
termtransform(ex) = (exs = Dict(); ex = termtransform(ex, exs); (Term(ex), exs))

function exprtransform(s::Symbolic, exs)
    return exs[s.name]
end
function exprtransform(ex::Expr, exs)
    if @capture(ex, f_(args__))
        return :($f($(map(arg->exprtransform(arg, exs), args)...)))
    else
        throw(ArgumentError("non expr transformable: $ex"))
    end
end

#=
macro @o(args...)
  args
end

macro @match(ex)
  :!
  quote
    for 
  end
end
function termtransform(T)
    #metatransform(expr, ::Type{typeof(+)}) = QuoteNode(+)
    if T isa DataType && isdefined(T, :instance)
        return @term($(T.instance))
    end
    return Symbolic(gensym(), TypeSet(T))
end

function termtransform(::Type{<:Broadcasted{Style, Axes, F, Args}}) where {Style, Axes, F, Args}
    f = termtransform(F)
    args = [termtransform(arg) for arg in Args.parameters]
    return @term(Broadcaster($f)($(args...)))
end

function termtransform(::Type{<:ArrayifiedArray{T, N, Arg}}) where {T, N, Arg}
    f = termtransform(F)
    arg = termtransform(Arg)
    return @term(ArrayifiedArray($f)($arg))
end

function termtransform(::Type{<:SwizzledArray{T, N, Arg, mask, Op}}) where {T, N, Arg, mask, Op}
    op = termtransform(Op)
    arg = termtransform(Arg)
    return @term(Swizzler($mask, $op)($arg))
end

function Recognizer()
  trs = broadcast_term_rewriting_system(Rewrite.rules())
  ctx = broadcast_context(Rewrite.DEFAULT_CONTEXT)
end


function _collapse(f, args)
    return Iterators.flatten(@capture(arg, g_(args′__)) && f(g) ? _collapse(f, args′) : (arg,) for arg in args)
end

function collapse(f, ex)
    println(ex)
    if @capture(ex, g_(arg_)) && f(g)
        return arg
    elseif @capture(ex, g_(args__)) && f(g)
        args′ = _collapse(f, args)
        return :($g($(_collapse(f, args)...)))
    else
        return ex
    end
end

collapse(f) = ex -> collapse(f, ex)

rules = [
    collapse(isequal(+)),
    collapse(isequal(*)),
    collapse(isequal(&)),
    collapse(isequal(|)),
    (ex -> @capture(ex, $(+)($(*)(~x_, ~y_), ~$(*)(~x_, ~z_), ~t_)) ? :($(+)($(*)($x, $(+)($(*)($(y...)), $(*)($(z...)))), $(t...))) : ex),
]


function normalize(ex, rules)
    ex′ = nothing
    while ex != ex′
        ex′ = ex
        postwalk(child->normalize(child, rules), ex)
        for rule in rules
            ex = rule(ex)
        end
    end
    return ex
end




    (ex -> @capture(ex, +(~(/(~x_, ~z_), /(~y_, ~z_), t__))) ? :(+(($x + $y)/$z, $(t...))) : ex),
    collapse(isequal(Broadcaster(+))),
    collapse(isequal(Broadcaster(*))),
    collapse(isequal(Broadcaster(&))),
    collapse(isequal(Broadcaster(|))),
        PatternRule(@term(x - y), @term(x + -y)),
        PatternRule(@term(x + -x), @term(zero(x))),
        PatternRule(@term(-(x + y)), @term(-y + -x)),
        PatternRule(@term(-(-x)), @term(x)),
        PatternRule(@term(-1 * x), @term(-x)),
        PatternRule(@term(-x * y), @term(-(x * y))),
        PatternRule(@term(x^0), @term(one(x))),
        PatternRule(@term(x^0.0), @term(one(x))),
        PatternRule(@term(x^1), @term(x)),
        PatternRule(@term(x^1.0), @term(x)),
        PatternRule(@term(abs(-x)), @term(abs(x))),
        PatternRule(@term(abs(x * y)), @term(abs(x) * abs(y))),
        PatternRule(@term(abs(inv(x))), @term(inv(abs(x)))),
        EvalRule(+),
        EvalRule(-),
        EvalRule(*),
        EvalRule(abs),
        EvalRule(&),
        EvalRule(|),
        EvalRule(!)
        Associative(+),
        Commutative(+),
        Associative(*),
        Commutative(*),
        Associative(&),
        Commutative(&),
        Associative(|),
        Commutative(|)
    ]
#using MacroTools

function broadcast_properties(props)
    props! = []
    for prop in props
        try
            prop! = broadcast_property(prop)
            if prop! != prop
                push!(props!, prop!)
            end
        catch ArgumentError
        end
    end
    props!
end

function broadcast_rules(rules)
    rules! = []
    for rule in rules
        try
            rule! = broadcast_rule(rule)
            if rule! != rule
                push!(rules!, rule!)
            end
        catch ArgumentError
        end
    end
    rules!
end

#SCRIPT 1
using Swizzle
using Rewrite
using InteractiveUtils

A = B = rand(3,3)

C = rand(3,3)

f(A, B, C) = (C .= Sum(2).(A.*Beam(2,3).(B)))

#display(@code_warntype(f(A, B, C)))
#println()
#println(typeof(Unwrap().(Sum(2).(A.+Beam(2,3).(B)))))
println(Swizzle.metatransform(:foo, typeof(Unwrap().(Sum(2).(2 .*Beam(2,3).(B))))))
(u, ts) = Swizzle.termtransform(Swizzle.metatransform(:foo, typeof(Unwrap().(Sum(2).(A.*Beam(2,3).(B))))))

println(u)

@syms a b c

#println(Rewrite.with_context(()->normalize(Term(u), Swizzle.LIBRARY.equalities), Swizzle.LIBRARY.context))
println(Rewrite.with_context(()->normalize(@term(a * b + a * c), Swizzle.LIBRARY.equalities), Swizzle.LIBRARY.context))

#SCRIPT 2
using Rewrite; using Swizzle
push!(Rewrite.CONTEXT.props, Flat(Swizzle.Broadcaster(+)))
push!(Rewrite.CONTEXT.props, Orderless(Swizzle.Broadcaster(+)))
rules = Rewrite.rules()
@vars a b c
foo = Unwrap().(rand(3,3).+(rand(3,3).+rand(3,3)))
myexpr = metatransform(:foo, typeof(foo))
println(myexpr)
println(match(@term($(Swizzle.Broadcaster(+))(a, b, c)), normalize(Term(myexpr), rules)))
println(normalize(Term(myexpr), rules))
myrule = Rewrite.PatternRule(@term($(Swizzle.Broadcaster(+))(a, b, c)), @term(println(a)))
println(myrule)
push!(rules.rules, myrule)
mynormterm = normalize(Term(myexpr), rules)
mynormexpr = get(mynormterm)
println(mynormterm)
println(mynormexpr)
Base.eval(Main, get(normalize(Term(myexpr), rules)))


#META jl

using Base.Broadcast: broadcasted, BroadcastStyle

struct Broadcaster{F}
  f::F
end

(bc::Broadcaster)(args...) = broadcasted(bc.f, args...)

Base.isequal(a::Broadcaster, b::Broadcaster) = a.f == b.f

function metatransform(expr, T)
    # essentially, this function performs `metatransform(expr, ::Type{typeof(+)}) = QuoteNode(+)`
    if T isa DataType && isdefined(T, :instance)
        return QuoteNode(T.instance)
    end
    return :($expr::$T)
end

function metatransform(expr, ::Type{<:Broadcasted{Style, Axes, F, Args}}) where {Style, Axes, F, Args}
    f = metatransform(:($expr.f), F)
    args = [metatransform(:($expr.args[$i]), arg) for (i, arg) in enumerate(Args.parameters)]
    if f isa QuoteNode
        return :($(Broadcaster(f.value))($(args...)))
    else
        return :((Broadcaster($expr.f))($(args...)))
    end
end

function metatransform(expr, ::Type{<:ArrayifiedArray{T, N, Arg}}) where {T, N, Arg}
    arg = metatransform(:($expr.arg), Arg)
    return :(ArrayifiedArray{$T}($arg))
end

function metatransform(expr, ::Type{<:SwizzledArray{T, N, Arg, mask, Op}}) where {T, N, Arg, mask, Op}
    op = metatransform(:($expr.op), Op)
    arg = metatransform(:($expr.arg), Arg)
    if op isa QuoteNode
        return :($(Swizzler(mask, op.value))($arg))
    else
        return :((Swizzler($mask, $expr.op))($arg))
    end
end

function metatransform(expr, ::Type{<:SwizzledArray{T, N, <:ArrayifiedArray{<:Any, <:Any, Arg}, mask, Op}}) where {T, N, Arg, mask, Op}
    op = metatransform(:($expr.op), Op)
    arg = metatransform(:($expr.arg.arg), Arg)
    if op isa QuoteNode
        return :($(Swizzler(mask, op.value))($arg))
    else
        return :((Swizzler($mask, $expr.op))($arg))
    end
end


=#
