# Functions that should be in Base that aren't
#     :o     so edgy...



Base.filter(f, ::Tuple{}) = ()
Base.filter(f, args::Tuple{Any}) = f(args[1]) ? args : ()
function Base.filter(f, args::Tuple)
    tail = filter(f, Base.tail(args))
    f(args[1]) ? (args[1], tail...) : tail
end



using Base.Broadcast: BroadcastStyle

struct Unwrap
  val
end

Unwrap() = Unwrap(nothing)

Base.Broadcast.broadcasted(::BroadcastStyle, ::Unwrap, bc) = Unwrap(bc)

Base.Broadcast.materialize(uw::Unwrap) = uw.val



using Base.Broadcast: Extruded

#Base.@propagate_inbounds Base.getindex(A::Extruded, I...) = Broadcast._broadcast_getindex(A, CartesianIndex(I))

#Base.@propagate_inbounds Base.getindex(A::Extruded, I) = Broadcast._broadcast_getindex(A, I)
