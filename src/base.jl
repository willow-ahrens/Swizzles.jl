# Functions that should be in Base that aren't
#     :o     so edgy...

using Base.Iterators
using Base: IteratorSize, HasShape, HasLength, IsInfinite

import Base.filter

filter(f, ::Tuple{}) = ()
filter(f, args::Tuple{Any}) = f(args[1]) ? args : ()
function filter(f, args::Tuple)
    tail = filter(f, Base.tail(args))
    f(args[1]) ? (args[1], tail...) : tail
end

import Base.getindex

add_iteratorsize(::Union{HasShape, HasLength}, ::Union{HasShape, HasLength}) = HasLength()
add_iteratorsize(::Union{HasShape, HasLength}, ::IsInfinite) = IsInfinite()
add_iteratorsize(::IsInfinite, ::Union{HasShape, HasLength}) = IsInfinite()
add_iteratorsize(::IsInfinite, ::IsInfinite) = IsInfinite()
add_iteratorsize(a, b) = SizeUnknown()

Base.IteratorSize(::Type{Flatten{Tuple{}}}) = HasLength
Base.IteratorSize(::Type{Flatten{Tuple{A}}}) where {A} = IteratorSize(A)
Base.IteratorSize(::Type{Flatten{Tuple{A, B}}}) where {A, B} = add_iteratorsize(IteratorSize(A), IteratorSize(B))

Base.getindex(a::Repeated, i::Int) = a.x

#=
Base.IteratorSize(::Type{Flatten{T}}) where {T <: Tuple} = mapfoldl(add_iteratorsize, map(IteratorSize, T.parameters))

function prepare_to_lift(it) where {I}
  if !haslength(fl.it)
    

end

function prepare_to_lift(fl::Flatten{I}) where {I}
  # If it is possible that there are an infinite number of iterators, we cannot
  # guarantee that this algorithm will terminate even if we know that fl is
  # infinite, as we may never reach the infinite iterator within fl.
  if !haslength(I)
    return fl
  else
    finite = []
    for i in fl.it
      thelength = Base.IteratorSize(i)
      if haslength(i)
        append!(finite, i)
      else
        return flatten(((finite...,), simplify_flatten(i)))
      end
    end
    return (finite)
  end
end
=#
