//This kernel performs reduction on an input array 

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);



int  __attribute__ ((noinline)) kernel_reduction_shared_mem(int *A, int N) {

	bsg_tile_group_shared_mem(int, sh_A, N);

	bsg_tile_group_shared_store (int, sh_A, bsg_id, A[bsg_id]);

	bsg_tile_group_barrier(&r_barrier, &c_barrier);

	int offset = 1;
	int mult = 2;

	while (offset < N) {
		if (!(bsg_id % mult)){
			int lc_A, lc_B;

			bsg_tile_group_shared_load (int, sh_A, bsg_id, lc_A);
			bsg_tile_group_shared_load (int, sh_A, bsg_id + offset, lc_B);

			bsg_tile_group_shared_store (int, sh_A, bsg_id, lc_A + lc_B);
		}

		bsg_tile_group_barrier (&r_barrier, &c_barrier);

		mult *= 2;
		offset *= 2;
	}

	if (bsg_id == 0)
		bsg_tile_group_shared_load(int, sh_A, 0, A[0]);

	bsg_tile_group_barrier (&r_barrier, &c_barrier);

  return 0;
}
