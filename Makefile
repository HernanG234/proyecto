IDIR = ./inc
CC=g++
CFLAGS=-I$(IDIR) -Wall `pkg-config opencv --cflags` -Wextra -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
LDFLAGS = `pkg-config opencv --libs` -lpthread
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE = test

DEPS = $(wildcard inc/*.h)

#_OBJ = ldb.o
#OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

src/%.o: src/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

$(EXECUTABLE): $(SOURCES) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.PHONY: clean

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
