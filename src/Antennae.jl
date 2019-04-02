module Antennae
using Swizzles.Properties

export Antenna

using Base.Broadcast: broadcasted

struct Antenna{F}
    f::F
end

(a::Antenna)(args...) = broadcasted(a.f, args...)

@inline Properties.initial(a::Antenna{F}, T) where {F} = Ref(initial(a.f, eltype(T)))
@inline Properties.initial(a::Antenna{F<:Intercept}, T) where {F} = initial(a.f, eltype(T))

end
