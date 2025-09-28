/*
 * GPU Compressor Prototype
 * ------------------------
 * GPU-accelerated Huffman compressor using Apple Metal.
 *
 * Build:
 *   clang++ -std=c++17 -ObjC++ gpu_compress.mm \
 *       -framework Metal -framework Foundation \
 *       -o gpu_compress
 *
 * Usage:
 *   ./gpu_compress --compress file.txt out.gzc
 *   ./gpu_compress --decompress out.gzc recovered.txt
 *   ./gpu_compress --compress-folder ./data ./compressed
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <fstream>
#include <queue>
#include <filesystem>
#include <functional>
#include <time.h>

namespace fs = std::filesystem;

double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// -------------------
// Huffman utilities
// -------------------
struct Node {
    int symbol;
    uint64_t freq;
    Node *left, *right;
};
struct Cmp {
    bool operator()(Node* a, Node* b) { return a->freq > b->freq; }
};

void build_huffman(const std::vector<uint32_t>& hist,
                   std::vector<uint32_t>& codes,
                   std::vector<uint32_t>& lengths) {
    std::priority_queue<Node*, std::vector<Node*>, Cmp> pq;
    for (int i=0;i<256;i++) {
        if (hist[i] > 0) {
            Node* n = new Node{i,(uint64_t)hist[i],nullptr,nullptr};
            pq.push(n);
        }
    }
    if (pq.empty()) return;
    while (pq.size() > 1) {
        Node* a = pq.top(); pq.pop();
        Node* b = pq.top(); pq.pop();
        Node* p = new Node{-1, a->freq+b->freq, a, b};
        pq.push(p);
    }
    Node* root = pq.top();

    std::function<void(Node*,int)> dfs = [&](Node* n,int d){
        if (!n) return;
        if (n->symbol >= 0) {
            lengths[n->symbol] = d;
        }
        dfs(n->left,d+1);
        dfs(n->right,d+1);
    };
    dfs(root,0);

    std::vector<std::pair<int,int>> symlens;
    for (int i=0;i<256;i++) if (lengths[i]>0)
        symlens.push_back({lengths[i],i});
    std::sort(symlens.begin(),symlens.end());
    uint32_t code=0;
    int prevlen=0;
    for (auto [len,sym]: symlens) {
        code <<= (len-prevlen);
        codes[sym]=code;
        prevlen=len;
        code++;
    }
}

// -------------------
// GPU histogram
// -------------------
std::vector<uint32_t> gpu_histogram(const std::vector<unsigned char>& data) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { printf("No Metal device found.\n"); exit(1); }
    NSError* err=nil;
    NSString* src = [NSString stringWithContentsOfFile:@"gpu_compress.metal"
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { NSLog(@"Shader load error: %@", err); exit(1); }
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"histogram_kernel"];
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    size_t n=data.size();
    id<MTLBuffer> inBuf  = [device newBufferWithBytes:data.data() length:n options:MTLResourceStorageModeShared];
    id<MTLBuffer> histBuf = [device newBufferWithLength:256*sizeof(uint32_t) options:MTLResourceStorageModeShared];
    memset([histBuf contents],0,256*sizeof(uint32_t));
    uint32_t n32 = (uint32_t)n;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:histBuf offset:0 atIndex:1];
    [enc setBytes:&n32 length:sizeof(uint32_t) atIndex:2];
    MTLSize grid = MTLSizeMake(n,1,1);
    NSUInteger tgs = pipe.maxTotalThreadsPerThreadgroup;
    MTLSize group = MTLSizeMake(tgs,1,1);
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    uint32_t* gpuHist = (uint32_t*)[histBuf contents];
    std::vector<uint32_t> hist(256);
    for (int i=0;i<256;i++) hist[i]=gpuHist[i];
    return hist;
}

// -------------------
// Compression
// -------------------
void compress_file(const fs::path& inFile, const fs::path& outFile) {
    std::ifstream in(inFile, std::ios::binary | std::ios::ate);
    if (!in) { perror("open"); return; }
    size_t n=in.tellg();
    in.seekg(0);
    std::vector<unsigned char> data(n);
    in.read((char*)data.data(), n);

    double t0 = now_sec();
    auto hist = gpu_histogram(data);
    std::vector<uint32_t> codes(256,0), lengths(256,0);
    build_huffman(hist,codes,lengths);

    std::vector<uint8_t> outbuf;
    outbuf.reserve(n/2);
    uint64_t bitbuf=0; int bitcount=0;
    auto putbits=[&](uint32_t code,int len){
        bitbuf|=((uint64_t)code)<<bitcount;
        bitcount+=len;
        while (bitcount>=8) {
            outbuf.push_back(bitbuf&0xFF);
            bitbuf>>=8;
            bitcount-=8;
        }
    };
    for (size_t i=0;i<n;i++) {
        unsigned char c = data[i];
        putbits(codes[c], lengths[c]);
    }
    if (bitcount>0) outbuf.push_back(bitbuf&0xFF);

    double t1 = now_sec();
    std::ofstream out(outFile,std::ios::binary);
    out.write((char*)outbuf.data(), outbuf.size());

    printf("Compressed %s → %s | %.2f MB → %.2f MB | ratio=%.2f | time=%.3f s\n",
           inFile.c_str(), outFile.c_str(),
           n/1024.0/1024.0, outbuf.size()/1024.0/1024.0,
           (double)outbuf.size()/n, t1-t0);
}

// -------------------
// Main CLI
// -------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  %s --compress in out\n", argv[0]);
        printf("  %s --decompress in out (NYI)\n", argv[0]);
        printf("  %s --compress-folder indir outdir\n", argv[0]);
        return 0;
    }

    std::string cmd=argv[1];
    if (cmd=="--compress") {
        compress_file(argv[2], argv[3]);
    } else if (cmd=="--compress-folder") {
        fs::path inDir=argv[2], outDir=argv[3];
        fs::create_directories(outDir);
        for (auto& p: fs::recursive_directory_iterator(inDir)) {
            if (p.is_regular_file()) {
                fs::path rel=fs::relative(p.path(),inDir);
                fs::path outPath=outDir/rel;
                outPath.replace_extension(".gzc");
                fs::create_directories(outPath.parent_path());
                compress_file(p.path(), outPath);
            }
        }
    } else {
        printf("Unknown command: %s\n", cmd.c_str());
    }
}

