-include ../makefile.inc

TESTSOBJDIR = bin
TESTS_HEADERS = $(wildcard *.h)
TESTS_SRC = $(wildcard *.cpp)
TESTS_OBJ = $(TESTS_SRC:%.cpp=$(TESTSOBJDIR)/%.o)
HEADER_INCL_DIR = -I./../src
LDFLAGS += -L./../src -lflinng
LIB_PATH = -L/usr/local/lib/ -lflinng
all: runme

runme: $(TESTSOBJDIR) $(TESTS_OBJ)
	$(CXX) -o $@ $(TESTS_OBJ) $(LDFLAGS) $(LIBS)

$(TESTS_OBJ): $(TESTSOBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPUFLAGS) $(LDFLAGS) $(HEADER_INCL_DIR) -c $< -o $@  

$(TESTSOBJDIR):   
	@ mkdir -p $(TESTSOBJDIR)

clean:
	rm -f runme
	rm -f $(TESTS_OBJ)
	rm -rf $(TESTSOBJDIR)

.PHONY: all clean runme
