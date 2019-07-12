#
# Makefile for non-Microsoft compilers
#

## Linux  (uncomment the 2 lines below for compilation on Linux)
CXXFLAGS += -std=c++98 -Wall -ggdb
LDFLAGS +=

## CygWin (uncomment the 2 lines below for compilation on CygWin)
#CXXFLAGS += -Wall
#LDFLAGS += 

MAIN = network-inference-multitree

all: $(MAIN) generate_nets

opt: CXXFLAGS += -O4
opt: LDFLAGS += -O4
opt: $(MAIN) generate_nets

linux: LDFLAGS += -lrt
linux: network-inference-multitree generate_nets

opt_linux: CXXFLAGS += -O4
opt_linux: LDFLAGS += -lrt -O4
opt_linux: network-inference-multitree generate_nets

# COMPILE
$(MAIN): $(MAIN).cpp Snap.o cascinf.cpp
	g++ -o network-inference-multitree $(MAIN).cpp cascinf.cpp Snap.o -I./glib -I./snap $(LDFLAGS) -lrt

generate_nets: generate_nets.cpp Snap.o cascinf.cpp
	g++ -o generate_nets generate_nets.cpp cascinf.cpp Snap.o -I./glib -I./snap $(LDFLAGS) -lrt
	
Snap.o: 
	g++ -c $(CXXFLAGS) ./snap/Snap.cpp -I./glib -I./snap -lrt

clean:
	rm -f   network-inference-multitree generate_nets Snap.o network-inference-multitree.exe generate_nets.exe
