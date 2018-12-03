using Base.Broadcast: BroadcastStyle

struct CompiledStyle{Style<:BroadcastStyle} <: BroadcastStyle
    style::Style
end

copyto_thunk(dest, desttype, src, srctype)

getindex_thunk(arr, arrtype, idx, idxtype)
