#
# Makefile for non-Microsoft compilers
#

## Linux  (uncomment the 2 lines below for compilation on Linux)
CXXFLAGS += -std=c++0x -Wall -ggdb
LDFLAGS += -lrt

## CygWin (uncomment the 2 lines below for compilation on CygWin)
##CXXFLAGS += -Wall
##LDFLAGS += 

MAIN = main

all: $(MAIN)

opt: CXXFLAGS += -O4
opt: LDFLAGS += -O4
opt: $(MAIN)

# COMPILE
$(MAIN): $(MAIN).cpp Snap.o 
	g++ $(LDFLAGS) -o main $(MAIN).cpp dani.cpp Snap.o -lrt -I./glib -I./snap -std=c++0x
Snap.o: 
	g++ -c $(CXXFLAGS) ./snap/Snap.cpp -lrt -I./glib -I./snap -std=c++0x

clean:
	rm -f   $(MAIN)
