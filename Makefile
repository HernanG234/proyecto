CC=g++

SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=test

CFLAGS = -Wall `pkg-config opencv --cflags`
LDFLAGS = `pkg-config opencv --libs`

all: $(EXECUTABLE)

debug: CFLAGS += -g -O0 -Wextra
debug: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCES) $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@ 

.cpp.o:
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)
