BASE = final

all: $(BASE)

OS := $(shell uname -s)

ifeq ($(OS), Linux)
  LIBS += -lGL -lGLU -lGLEW -lglfw
endif

ifeq ($(OS), Darwin)
  CPPFLAGS += -D__MAC__ -std=c++11 -stdlib=libc++
  LDFLAGS += -framework OpenGL -framework IOKit -framework Cocoa
  LIBS += -lglfw.3 -lGLEW
endif

ifdef OPT
  #turn on optimization
  CXXFLAGS += -O2
else
  #turn on debugging
  CXXFLAGS += -g
endif

CXX = g++

OBJ = $(BASE).o ppm.o glsupport.o scenegraph.o picker.o geometry.o material.o renderstates.o texture.o

$(BASE): $(OBJ)
	$(LINK.cpp) -o $@ $^ $(LIBS)

clean:
	rm -f $(OBJ) $(BASE)
