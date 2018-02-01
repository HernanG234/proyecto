IDIR = ./inc
CC=g++
CFLAGS=-I$(IDIR) -Wall `pkg-config opencv --cflags`
LDFLAGS = `pkg-config opencv --libs`
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE = test

LDFLAGS = `pkg-config opencv --libs`

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
