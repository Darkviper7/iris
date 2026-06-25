ROCM ?= /opt/rocm-7.2.4

tcc_tool.so: tcc_tool.cpp
	g++ -shared -fPIC -std=c++17 -D__HIP_PLATFORM_AMD__ \
	    -I$(ROCM)/include \
	    tcc_tool.cpp \
	    -L$(ROCM)/lib -lrocprofiler-sdk -Wl,-rpath,$(ROCM)/lib \
	    -o tcc_tool.so

clean:
	rm -f tcc_tool.so

.PHONY: clean
