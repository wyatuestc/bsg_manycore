//This kernel performs a barrier among all tiles in tile group N times

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"


#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);


int  __attribute__ ((noinline)) kernel_barrier_loop(int N) {

	bsg_cuda_print_stat_kernel_start();

	for (int i = 0; i < N; i ++) {
		bsg_cuda_print_stat_start(1);
		bsg_tile_group_barrier(&r_barrier, &c_barrier);
		bsg_cuda_print_stat_end(1);
	}

	bsg_cuda_print_stat_kernel_end();

return 0;
}
