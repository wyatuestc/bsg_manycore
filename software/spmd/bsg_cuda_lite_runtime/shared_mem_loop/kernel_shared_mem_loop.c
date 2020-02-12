//This kernel performs a load store into tile group shared memory N times

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"


#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);


int  __attribute__ ((noinline)) kernel_shared_mem_loop(int N) {

	bsg_cuda_print_stat_kernel_start();

	bsg_tile_group_shared_mem (int, sh_A, N);
        int local_A [N];

	bsg_cuda_print_stat_start(1);
	// Load entire shared memory into local memory
	for (int i = 0; i < N; i ++) {
		bsg_tile_group_shared_load (int, sh_A, i, local_A[i]);
	}
	bsg_cuda_print_stat_end(1);


	bsg_tile_group_barrier(&r_barrier, &c_barrier);


	bsg_cuda_print_stat_start(2);
	// Store entire local memory back into shared memory
	for (int i = 0; i < N; i ++) {
		bsg_tile_group_shared_store (int, sh_A, i, local_A[i]);
	}
	bsg_cuda_print_stat_end(2);

	bsg_cuda_print_stat_kernel_end();

return 0;
}
