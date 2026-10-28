# fixed rays pt 0 with pp
./GPU_Benchmark 0 1 1
./CPU_Benchmark 0 1

# fixed rays pt 0 without pp
./GPU_Benchmark 0 1 0

# fixed rays pt 1 with pp
./GPU_Benchmark 1 1 1
./CPU_Benchmark 1 1 

# fixed rays pt 1 without pp
./GPU_Benchmark 1 1 0

# rays per point pt 0 with pp
./GPU_Benchmark 0 0 1
./CPU_Benchmark 0 0

# rays per point pt 0 without pp
./GPU_Benchmark 0 0 0

# rays per point pt 1 with pp
./GPU_Benchmark 1 0 1
./CPU_Benchmark 1 0 

# rays per point pt 1 without pp
./GPU_Benchmark 1 0 0
