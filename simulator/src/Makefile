
# import configuration file
include configuration

# fortran compiler
FC     = $(FC_$(COMPILER))
FFLAGS = $(FFLAGS_$(COMPILER))
TARCH  = $(TARGET_$(COMPILER))
# omp compiler
OMP_NO  =
OMP_YES = $($(COMPILER)_OMP)

# commands
COMPILE = $(FC) $(FFLAGS) $(OMP_$(PARALL)) $(TARCH)

# programs
F90DIR  = f90
F90S    = $(wildcard $(F90DIR)/*.f90)
# objects
OBJSDIR = obj
OBJS    = $(F90S:$(F90DIR)/%.f90=$(OBJSDIR)/%.o)
# executables
BINDIR  = ../bin
EXEC    = $(BINDIR)/model

all: $(EXEC) clean

$(EXEC): $(OBJS)
	$(FC) $(FFLAGS) $(TARCH) $(OMP_$(PARALL)) $^ -o $(EXEC)

$(OBJSDIR)/%.o: $(F90DIR)/%.f90
	$(COMPILE) -c $< -o $@

dist: clean
	rm -f $(EXEC)

clean:
	rm -f $(OBJS)
	rm -f *.mod
	rm -f ./f90/*.mod
	rm -f *.o
	
include dependencies 
