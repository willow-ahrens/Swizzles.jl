module Antennae
using Swizzles.Properties

export Antenna

using Base.Broadcast: broadcasted

struct Antenna
    f
end

(a::Antenna)(args...) = broadcasted(a.f, args...)

@inline Properties.initial(a::Antenna, T) = Ref(initial(a.f, eltype(T)))

end
