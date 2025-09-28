#include <metal_stdlib>
using namespace metal;

kernel void histogram_kernel(
    device const uchar *input      [[ buffer(0) ]],
    device atomic_uint *histogram  [[ buffer(1) ]],
    constant uint &n               [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    uchar c = input[gid];
    atomic_fetch_add_explicit(&histogram[c], 1u, memory_order_relaxed);
}

