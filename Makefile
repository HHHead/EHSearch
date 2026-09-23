# Makefile - EHSearch reference implementation
#
# Targets
#   make            build the benchmark driver and all auxiliary tools
#   make main       build only the benchmark driver   -> build/ehsearch
#   make tools      build the auxiliary tools         -> build/<tool>
#   make clean      remove build artefacts
#
# The sources are x86-64 oriented: they use SSE/AVX/AVX-512 intrinsics and the
# AES-NI / VAES AES intrinsics.  Runtime dispatch (see simd.h) selects the widest
# instruction set actually available on the host, but the translation units still
# have to be compiled with the corresponding ISA flags, which is why -maes and
# -march=native are on by default.

CXX      ?= g++
CXXSTD   ?= -std=c++17
OPT      ?= -O3 -march=native -funroll-loops -DNDEBUG
OMP      ?= -fopenmp
AES      ?= -maes
INCLUDES ?= -Isrc
CXXFLAGS ?= $(CXXSTD) $(OPT) $(OMP) $(AES) $(INCLUDES)

SRC_DIR   := src
BUILD_DIR := build
BIN       := $(BUILD_DIR)/ehsearch

CORE_SRC  := $(SRC_DIR)/main.cpp $(SRC_DIR)/kernels.cpp
HEADERS   := $(wildcard $(SRC_DIR)/*.h)
TOOLS     := diag_dist diag_dump diag_graph diag_search test_hnsw test_plain tune_hnsw tune_scale

# MinGW links the process-memory API explicitly; POSIX systems do not need it.
ifeq ($(OS),Windows_NT)
  PLATFORM_LIBS ?= -lpsapi
else
  PLATFORM_LIBS ?=
endif

.PHONY: all main tools clean
.SUFFIXES:

all: main tools

main: $(BIN)

$(BIN): $(CORE_SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CORE_SRC) -o $@ $(LDFLAGS) $(PLATFORM_LIBS)

tools: $(TOOLS:%=$(BUILD_DIR)/%)

$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(PLATFORM_LIBS)

clean:
	rm -rf $(BUILD_DIR)
