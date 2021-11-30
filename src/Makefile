-include ../makefile.inc

HEADERS     = $(wildcard *.h)
CPPOBJDIR = bin
CPPSOURCES := $(wildcard *.cpp)
CPPOBJS := $(CPPSOURCES:%.cpp=$(CPPOBJDIR)/%.o)

prefix      ?= /usr/local
exec_prefix ?= ${prefix}
libdir       = ${exec_prefix}/lib
includedir   = ${prefix}/include

.PHONY: all clean install installdirs uninstall

############################

all: libflinng.a libflinng.$(SHAREDEXT)
	@echo ""
	@echo "Flinng lib is ready!!"
############################

libflinng.a: $(CPPOBJDIR) $(CPPOBJS)
	@echo ""
	@echo "##### Creating $@ #####"
	$(AR) r $@ $(CPPOBJS)

libflinng.$(SHAREDEXT): $(CPPOBJDIR) $(CPPOBJS)
	@echo ""
	@echo "##### Creating $@ #####"
	$(CXX) $(SHAREDFLAGS) $(LDFLAGS) -o $@ $(CPPOBJS)

############################

$(CPPOBJS): $(CPPOBJDIR)/%.o: %.cpp
	@echo ""
	@echo "##### Compiling $@ $< #####"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPPOBJDIR):   
	@ mkdir -p $(CPPOBJDIR)
############################

clean:
	$(RM) -rf libflinng.a libflinng.$(SHAREDEXT)
	$(RM) -rf $(CPPOBJDIR)

############################
# Installing

install: $(INSTALLDIRS) libflinng.a libflinng.$(SHAREDEXT) 
	mkdir -p $(DESTDIR)$(includedir)/flinng
	cp libflinng.a $(DESTDIR)$(libdir)
	cp libflinng.$(SHAREDEXT) $(DESTDIR)$(libdir)
	tar cf - $(HEADERS) | tar xf - -C $(DESTDIR)$(includedir)/flinng/

$(INSTALLDIRS):
	mkdir -p $(DESTDIR)$(includedir)/flinng


uninstall:
	$(RM) -f $(DESTDIR)$(libdir)/libflinng.a \
	      $(DESTDIR)$(libdir)/libflinng.$(SHAREDEXT)
	$(RM) -rf $(DESTDIR)$(includedir)/flinng
############################
