//This kernel performs reduction on an input array 

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);



int  __attribute__ ((noinline)) kernel_reduction_shared_mem(int *A, int N) {

	int sum = 0;
	for (int i = 0; i < N; i ++) 
		sum += A[i];
	A[0] = sum;

	bsg_tile_group_barrier(&r_barrier, &c_barrier); 

  return 0;
}
