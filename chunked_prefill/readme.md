nvcc -shared -o  chunked_prefill.so  chunked_prefill.cu  -lcudart -arch=sm_80  -Xcompiler -fPIC
