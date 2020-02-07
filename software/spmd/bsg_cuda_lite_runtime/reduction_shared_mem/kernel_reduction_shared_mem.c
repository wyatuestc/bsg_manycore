//This kernel performs reduction on an input array 

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);



int  __attribute__ ((noinline)) kernel_reduction_shared_mem(int *A, int N) {

	bsg_tile_group_shared_mem (int, sh_A, N);

	for (int iter_x = bsg_x; iter_x < N; iter_x += bsg_tiles_X * bsg_tiles_Y) {
		bsg_tile_group_shared_store (int, sh_A, (iter_x), A[iter_x]);
	}

	bsg_tile_group_barrier(&r_barrier, &c_barrier);

	int sum = 0;
	for (int iter_x = bsg_x; iter_x < N; iter_x += bsg_tiles_X * bsg_tiles_Y) {
		int lc_A;
		bsg_tile_group_shared_load (int, sh_A, (iter_x), lc_A);
		sum += lc_A;
	}
	
	bsg_tile_group_barrier(&r_barrier, &c_barrier);

	A[0] = sum;

	bsg_tile_group_barrier(&r_barrier, &c_barrier); 

  return 0;
}
