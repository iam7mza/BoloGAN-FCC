CXX      = g++

EDM4HEP_BASE = /cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/edm4hep/1.0-kavkp3
EDM4HEP_INC  = $(EDM4HEP_BASE)/include
EDM4HEP_LIB  = $(EDM4HEP_BASE)/lib64

PODIO_BASE = /cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/podio/1.7-i3s63v
PODIO_INC  = $(PODIO_BASE)/include
PODIO_LIB  = $(PODIO_BASE)/lib64

CXXFLAGS = $(shell root-config --cflags) -I$(EDM4HEP_INC) -I$(PODIO_INC)
LIBS     = $(shell root-config --libs) \
           -L$(EDM4HEP_LIB) -ledm4hep -Wl,-rpath,$(EDM4HEP_LIB) \
           -L$(PODIO_LIB)   -lpodio -lpodioIO -lpodioRootIO -Wl,-rpath,$(PODIO_LIB)

TARGET   = extractFromROOT
SRCS     = extractFromROOT.cxx

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
