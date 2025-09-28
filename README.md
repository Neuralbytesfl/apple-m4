# GPU Compressor Prototype

A Metal-accelerated compressor that uses GPU histograms and Huffman coding to achieve high throughput on Apple Silicon.

## Build


clang++ -std=c++17 -ObjC++ gpu_compress.mm \
  -framework Metal -framework Foundation \
  -o gpu_compress

