# Functions that should be in Base that aren't
#     :o     so edgy...

import Base.filter

filter(f, ::Tuple{}) = ()
filter(f, args::Tuple{Any}) = f(args[1]) ? args : ()
function filter(f, args::Tuple)
    tail = filter(f, Base.tail(args))
    f(args[1]) ? (args[1], tail...) : tail
end
