# Swizzles.jl

## Overview

`Swizzles` is an interface to array operations written in Julia. We
designed `swizzles` to function harmoniously with Julia's `broadcast` facilities, so that users can use (mostly) familiar syntax to build their own tensor kernels in a high-level language.

For developers, `Swizzles` is an experimental approach to generating efficient fused multidimensional broadcast and reduction operations. Swizzles provides the lazy `SwizzledArray` type, which represents a lazy simultaneous reduction and transposition, and the `GeneratedArray` abstract type, which users can override to take advantage of a fused trait-based implementation of Julia's `AbstractArray` built on lazy array types like `Broadcasted`, `SubArray`, and `SwizzledArray`.
