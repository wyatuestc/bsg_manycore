
bsg_tiles_X = 3
bsg_tiles_Y = 3


all: main.run

include ../Makefile.include


OBJS= main.o bsg_set_tile_x_y.o

main.riscv:  $(OBJS) $(SPMD_COMMON_OBJECTS) ../common/crt.o 
	$(RISCV_LINK)  $(OBJS) -o $@ $(RISCV_LINK_OPTS)

clean:
	-rm -rf *.o *.jou *.log *.pb bsg_manycore_io_complex_rom.v main.riscv *.wdb main.bin main.hex
	-rm -rf xsim.dir
	-rm -rf ./simv csrc simv.daidir ucli.key DVEfiles *.vpd

main.o: Makefile

include ../../mk/Makefile.tail_rules
