
AR ?= ar
CXX = g++
CC = gcc
RM = rm

INC := /usr/include/
LIB := -L/usr/lib64/
LIB += -fopenmp

OPT_FLAGS := -Ofast -fopenmp -march=native -fPIC
CXXFLAGS := -m64 -DUNIX -std=c++11 -g -fno-omit-frame-pointer $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)
CFLAGS := -m64 -DUNIX $(WARN_FLAGS) $(OPT_FLAGS) -I$(INC)
LDFLAGS := $(LIBRARY_PATH) $(LIB)

SHAREDEXT   = so
SHAREDFLAGS = -shared