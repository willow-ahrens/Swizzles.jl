using SpecialSets
using Rewrite
using Rewrite: PatternRule, EvalRule, Rule, Rules, Associative, Commutative, Property
using MacroTools

broadcast_term(t::Term) = Term(broadcast_term(get(t)))
broadcast_term(t) = t
function broadcast_term(ex::Expr)
    if ex.head == :call
        f, args = ex.args[1], ex.args[2:end]
        if f isa Union{Symbolic, Variable} # return f(b_t(arg1), b_t(arg2), b_t(arg3), ...)
            return Expr(:call, f, map(broadcast_term, args)...)
        elseif f isa Expr
            throw(ArgumentError("nonbroadcastable term: $ex"))
        elseif f isa Symbol
            throw(ArgumentError("nonbroadcastable term: $ex"))
        else # return Broadcaster(f)(b_t(arg1), b_t(arg2), b_t(arg3))
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
