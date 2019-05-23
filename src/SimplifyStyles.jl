module SimplifyStyles

using Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted, Style, AbstractArrayStyle, DefaultArrayStyle, result_style

using LinearAlgebra
using MacroTools
using Rewrite
using Rewrite: Rule, PatternRule, Property

using Swizzles
using Swizzles.Antennae
using Swizzles.ArrayifiedArrays
using Swizzles.ExtrudedArrays
using Swizzles.NamedArrays
using Swizzles.Properties
using Swizzles.ValArrays
using Swizzles.Virtuals
using Swizzles: Swizzle, SwizzledArray, mask, masktuple, imasktuple

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

function rewriteable(root, ::Type{<:NamedArray{<:Any, <:Any, Arr, name}}, syms) where {Arr, name}
    s = Symbolic(name)
    syms[s] = :($root.arg)
    return :($s::$Arr)
end

function rewriteable(root, ::Type{<:Adjoint{<:Any, Arg}}, syms) where Arg
    arg = rewriteable(:($root.parent), Arg, syms)
    :($Adjoint($arg))
end

function rewriteable(root, ::Type{<:Transpose{<:Any, Arg}}, syms) where Arg
    arg = rewriteable(:($root.parent), Arg, syms)
    :($Transpose($arg))
end

function rewriteable(root, T::Type{<:Broadcasted{<:Any, <:Any, F, Args}}, syms) where {F, Args<:Tuple}
    args = map(((i, arg),) -> rewriteable(:($root.args[$i]), arg, syms),
               enumerate(Args.parameters))
    some_f = instance(F)
    if !isnothing(some_f)
        f = something(some_f)
        :($Antenna($f)($(args...)))
    else
        return :($root::T)
    end
end

function rewriteable(root, T::Type{<:SwizzledArray{<:Any, <:Any, Op, _mask, Init, Arg}}, syms) where {Op, _mask, Init, Arg}
    init = rewriteable(:($root.init), Init, syms)
    arg = rewriteable(:($root.arg), Arg, syms)
    some_op = instance(Op)
    if !isnothing(some_op)
        op = something(some_op)
        :($Swizzle($op, $_mask)($init, $arg))
    else
        return :($root::T)
    end
end

function rewriteable(root, ::Type{<:ArrayifiedArray{<:Any, <:Any, Arg}}, syms) where {Arg}
    rewriteable(:($root.arg), Arg, syms)
end

"""
Utility function for making a rewritten term nicer before printing.
"""
prettify(t::Term) = Term(prettify(t.ex))

function prettify(ex)
    if @capture(ex, f_(args__))
        pargs = map(prettify, args)
        if f === Swizzle
            return Expr(:Swizzle, pargs...)
        elseif f === Antenna
            return Expr(:Antenna, pargs...)
        end
        return Expr(:call, prettify(f), pargs...)
    elseif @capture(ex, x_::t_)
        return prettify(x)
    end
    return ex
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

struct UndefinedSyms end
evaluable(ex) = evaluable(ex, UndefinedSyms())
Base.getindex(::UndefinedSyms, sym) = gensym()

"""
Virtually evaluate a Term (used for Rewrite.jl)
"""
function veval(ex::Union{Expr, Symbol})
    if @capture(ex, x_::T_)
        @assert T isa Type
        return virtualize(x, T)
    elseif @capture(ex, f_(args__))
        return veval(f)(map(veval, args)...)
    else
        throw(ArgumentError("Cannot virtually evaluate expression $ex"))
    end
end
veval(x) = x



"""
Stores rules for rewriting.
"""
struct RewriteSpec
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
            return Expr(:call, :($Antenna($f)), map(antenna_expr, args)...)
        end
    end
    throw(ArgumentError("can't convert to antenna version: $ex"))
end


"""
Helper function that converts a vanilla rule to one that uses Antennas.
"""
function antenna_rule(rule::PatternRule) :: PatternRule
    if !isempty(rule.ps)
        throw(ArgumentError("can't convert rule with properties: $r"))
    end
    return PatternRule(antenna_term(rule.left), antenna_term(rule.right))
end

struct ArbRule <: Rule
    attempt_transform
end
Rewrite.normalize(term::Term, arule::ArbRule) = arule.attempt_transform(term)

# Build RewriteSpec for normalization.
NORMALIZE_SPEC = begin
    @vars x y z

    pointwise_rules = Array{Rule, 1}([
        PatternRule(@term(x * (y + z)), @term(x * y + x * z))
    ])

    antenna_rules = map(antenna_rule, pointwise_rules)
    append!(antenna_rules, Array{Rule, 1}([
        PatternRule(@term( Antenna(identity)(x) ), @term( x ))
    ]))

    swizzle_rules = Array{Rule, 1}([
        # Remove zero and nothing inits
        ArbRule(term -> begin
            @vars _op _mask _init _arr
            for σ in match(@term(Swizzle(_op, _mask)(_init, _arr)), term)
                op, mask, init, arr = map(
                    _v -> σ[_v], (_op, _mask, _init, _arr))

                init isa ValArray || continue
                init_val = init[]
                T = eltype(veval(evaluable(arr)))

                isnothing(init_val) || Some(init_val) === initial(op, T) || continue
                return @term( Swizzle(op, mask)(arr) )
            end
            return term
        end),

        # Collapse nested Swizzles
        ArbRule(term -> begin
            @vars _op1 _mask1 _op2 _mask2 _arr
            @vars _init1 # optional
            for σ in union(
                match(@term( Swizzle(_op1, _mask1)(_init1, Swizzle(_op2, _mask2)(_arr)) ), term),
                match(@term( Swizzle(_op1, _mask1)(        Swizzle(_op2, _mask2)(_arr)) ), term)
            )
                op1, mask1, op2, mask2, arr = map(
                    _v -> σ[_v], (_op1, _mask1, _op2, _mask2, _arr))

                isnothing(op1) || isnothing(op2) || op1 === op2 || continue
                op = isnothing(op1) ? op2 : op1
                mask′ = masktuple(i -> nil, i -> mask2[i], mask1)

                if haskey(σ, _init1)
                    init1 = σ[_init1]
                    return @term( Swizzle(op, mask′)(init1, arr) )
                else
                    return @term( Swizzle(op, mask′)(arr) )
                end
            end
            return term
        end)
    ])

    rules = vcat(pointwise_rules, antenna_rules, swizzle_rules)
    RewriteSpec(Rules(rules), Context([]))
end

"""
    normalize(root, T::Type) :: Tuple{Term, Dict}

Normalizes the rewriteable expr generated from arr, using root as the root expr.
Returns the normalized Term and the syms Dict used to make the Term evaluable.
"""
function normalize(root, T::Type)
    term, syms = rewriteable(root, T)
    normal_term = Rewrite.with_context(NORMALIZE_SPEC.context) do
        Rewrite.normalize(term, NORMALIZE_SPEC.rules)
    end
    return normal_term, syms
end

"""
    reindex_masks(masks...) :: Tuple

A helper function that reindexes a list of masks in a canonical fashion.
# Examples
```jldoctest
julia> reindex_masks((100, 300), (200, 300))
((1, 2), (3, 2))
```
"""
function reindex_masks(masks...) :: Tuple
    idx_dict = Dict{Any, Int}()
    get_idx(key) = haskey(idx_dict, key) ?
                        idx_dict[key] : (idx_dict[key] = length(idx_dict) + 1)

    map(_mask -> map(get_idx, _mask), masks)
end

"""
    get_imasks_for_gemm_match(T1, T2, mask_out, mask_in1, mask_in2)

Returns a normalized tuple of (mask_out, mask_in1, mask_in2) if types and masks
are of the correct format -- otherwise returns nothing.
"""
function normalize_masks(T1, T2, mask_out, mask_in1, mask_in2)
    any([
        (   T1 <: ExtrudedArray{et, 2, <:Any, <:Any}
            && T2 <: ExtrudedArray{et, 2, <:Any, <:Any})
        for et in [Float64, Float32, ComplexF64, ComplexF32]
    ]) || return

    # Check dimensions of masks are ok.
    (length(mask_out) == 2 &&
        findfirst(isequal(2), mask_in1) !== nothing &&
        findfirst(isequal(2), mask_in2) !== nothing &&
        findfirst(isequal(3), mask_in1) === nothing &&
        findfirst(isequal(3), mask_in2) === nothing) || return

    imask1 = imasktuple(_ -> nil, identity, mask_in1, 2)
    imask2 = imasktuple(_ -> nil, identity, mask_in2, 2)

    rmasks = reindex_masks(mask_out, imask1, imask2)
    return rmasks
end

"""
    COPY_MATCHERS

An array of functions which take in a normalized array term destined for a copy
and returns either Some(matched_kernel_expression) or nothing.
"""
COPY_MATCHERS = Array{Function, 1}([
    # gemm(tA, tB, A, B)
    term -> begin
        @vars _mask_out _mask_in1 _mask_in2 _arr1 _T1 _arr2 _T2
        for σ in match(
            @term(
                Swizzle(+, _mask_out)(
                    Antenna(*)(
                        Swizzle(nothing, _mask_in1)(_arr1::_T1),
                        Swizzle(nothing, _mask_in2)(_arr2::_T2)
                    )
                )
            ),
            term
        )
            mask_out, mask_in1, mask_in2, arr1, arr2, T1, T2 = map(
                _v -> σ[_v],
                (_mask_out, _mask_in1, _mask_in2, _arr1, _arr2, _T1, _T2))

            nmasks = normalize_masks(T1, T2, mask_out, mask_in1, mask_in2)
            !isnothing(nmasks) || continue
            if minimum(nmasks[2]) > 1
                nmasks = (nmasks[1], nmasks[3], nmasks[2])
                tmp = arr1; arr1 = arr2; arr2 = tmp
            end

            gemm_patterns = Dict(
                ((1, 2), (1, 3), (3, 2)) => @term( BLAS.gemm('N', 'N', parent(arr1), parent(arr2)) ),
                ((1, 2), (3, 1), (3, 2)) => @term( BLAS.gemm('T', 'N', parent(arr1), parent(arr2)) ),
                ((1, 2), (1, 3), (2, 3)) => @term( BLAS.gemm('N', 'T', parent(arr1), parent(arr2)) ),
                ((1, 2), (3, 1), (2, 3)) => @term( BLAS.gemm('T', 'T', parent(arr1), parent(arr2)) )
            )
            haskey(gemm_patterns, (nmasks)) || continue

            @debug "matched gemm(tA, tB, A, B)"
            return gemm_patterns[nmasks]
        end
    end
])

"""
    COPYTO_MATCHERS

An array of functions which take in a normalized array term destined for a
copyto! and returns either Some(matched_kernel_expression) or nothing.
"""
COPYTO_MATCHERS = Array{Function, 1}([
    # gemm!(tA, tB, 1, A, B, 0, C)
    term -> begin
        @vars _mask_out _mask_in1 _mask_in2 _arr1 _T1 _arr2 _T2
        for σ in match(
            @term(
                Swizzle(+, _mask_out)(
                    Antenna(*)(
                        Swizzle(nothing, _mask_in1)(_arr1::_T1),
                        Swizzle(nothing, _mask_in2)(_arr2::_T2)
                    )
                )
            ),
            term
        )
            mask_out, mask_in1, mask_in2, arr1, arr2, T1, T2 = map(
                _v -> σ[_v],
                (_mask_out, _mask_in1, _mask_in2, _arr1, _arr2, _T1, _T2))

            nmasks = normalize_masks(T1, T2, mask_out, mask_in1, mask_in2)
            !isnothing(nmasks) || continue
            if minimum(nmasks[2]) > 1
                nmasks = (nmasks[1], nmasks[3], nmasks[2])
                tmp = arr1; arr1 = arr2; arr2 = tmp
            end

            eltype = T1.parameters[1]
            dst = Symbolic(DST_SYMBOL)
            gemm_patterns = Dict(
                ((1, 2), (1, 3), (3, 2)) => @term( BLAS.gemm!('N', 'N', eltype(1), parent(arr1), parent(arr2), eltype(0), dst) ),
                ((1, 2), (3, 1), (3, 2)) => @term( BLAS.gemm!('T', 'N', eltype(1), parent(arr1), parent(arr2), eltype(0), dst) ),
                ((1, 2), (1, 3), (2, 3)) => @term( BLAS.gemm!('N', 'T', eltype(1), parent(arr1), parent(arr2), eltype(0), dst) ),
                ((1, 2), (3, 1), (2, 3)) => @term( BLAS.gemm!('T', 'T', eltype(1), parent(arr1), parent(arr2), eltype(0), dst) )
            )
            haskey(gemm_patterns, nmasks) || continue

            @debug "matched gemm!(tA, tB, alpha, A, B, beta, C)"
            return gemm_patterns[nmasks]
        end
    end
])

"""
    match_term(term::Term, dst::Union{Type{<:Some}, Type{Nothing}}, syms)

Tries to match the term. Returns nothing if not possible.
"""
function match_term(term::Term, dst::Union{Type{<:Some}, Type{Nothing}}, syms)
    matchers = (dst <: Nothing) ? COPY_MATCHERS : COPYTO_MATCHERS
    for matcher in matchers
        matched_term = matcher(term)
        if !isnothing(matched_term)
            matched_expr = evaluable(matched_term, syms)
            return matched_expr
        end
    end
end

"""
    simplify_and_copy(arr, has_dst)

Does the following:

    1. Apply rules to normalize the array expression `arr`

    2. Try to match the normalized expression against a repository of fast
       kernels.

    3. Execute the matched implementation or copy the normalized expression.
       Depending on whether dst is something or nothing, copyto! or copy will
       get called respectively.
"""
@generated function simplify_and_copy(arr, dst::Union{Some{<:Any}, Nothing})
    normal_term, syms = normalize(:arr, arr)
    syms[Symbolic(DST_SYMBOL)] = :(something($DST_SYMBOL))

    matched_expr = match_term(normal_term, dst, syms)
    if !isnothing(matched_expr)
        return matched_expr
    end

    normal_expr = evaluable(normal_term, syms)
    return ((dst <: Nothing) ?
                :(copy($normal_expr))
              : :(copyto!(something(dst), $normal_expr)))
end


"""
    Simplify

Simplify is an abstract type which when passed as the first argument of
broadcasted returns the second argument stylized with a SimplifyStyle.

On copy, a Broadcasted with a SimplifyStyle will be simplified. Simplification
consists of two stages:

    1. The expression in question is converted into normal form.

    2. The normalized expression is matched against a repository of fast
       implementations. If no fast implementation is found, the normalized
       expression in executed.

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
struct SimplifyStyle{S<:BroadcastStyle} <: BroadcastStyle end

SimplifyStyle(style::S) where {S <: BroadcastStyle} = SimplifyStyle{S}()
SimplifyStyle(style::S) where {S <: SimplifyStyle} = style
Base.Broadcast.BroadcastStyle(::SimplifyStyle{T}, ::S) where {T, S<:AbstractArrayStyle} = SimplifyStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::SimplifyStyle{T}, ::S) where {T, S<:DefaultArrayStyle} = SimplifyStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::SimplifyStyle{T}, ::S) where {T, S<:Style{Tuple}} = SimplifyStyle(result_style(T(), S()))
Base.Broadcast.BroadcastStyle(::SimplifyStyle{T}, ::SimplifyStyle{S}) where {T, S<:BroadcastStyle} = SimplifyStyle(result_style(T(), S()))

function Base.Broadcast.broadcasted(::Simplify, b::Broadcasted{S}) where {S}
    Broadcasted{SimplifyStyle{S}}(b.f, b.args)
end
function Base.Broadcast.broadcasted(::Simplify, x)
    broadcasted(Simplify(), broadcasted(identity, x))
end

Base.@propagate_inbounds function Base.copy(bc::Broadcasted{SimplifyStyle{S}}) where {S <: AbstractArrayStyle}
    bc = Broadcasted{S}(bc.f, bc.args)
    lbc = bc |> lift_vals |> lift_keeps |> lift_names
    simplify_and_copy(lbc, nothing)
end

DST_SYMBOL = :dst
Base.@propagate_inbounds function Base.copyto!(dst::AbstractArray,
                                               src::Broadcasted{<:SimplifyStyle{S}}) where {S <: AbstractArrayStyle}
    bc = Broadcasted{S}(src.f, src.args)
    sbc = lift_names(bc |> lift_vals |> lift_keeps,
                     IdDict{Any, Any}(dst => DST_SYMBOL))
    simplify_and_copy(sbc, Some(dst))
end

end
