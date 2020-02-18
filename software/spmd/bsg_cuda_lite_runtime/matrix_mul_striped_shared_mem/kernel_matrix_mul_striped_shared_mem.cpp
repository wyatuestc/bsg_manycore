/*!
 * This kernel performs tiled matrix multiplication with use of tile-group-shared memory
 */

#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_striped_array.hpp"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);


using namespace bsg_manycore;


#define BLOCK_WIDTH 4
#define block_size_y 4
#define block_size_x 4


using Array_A = TileGroupStripedArray<int, (block_size_y * BLOCK_WIDTH),  bsg_tiles_X, bsg_tiles_Y, 1>;
using Array_B = TileGroupStripedArray<int, (BLOCK_WIDTH  * block_size_x), bsg_tiles_X, bsg_tiles_Y, 1>;
using Array_C = TileGroupStripedArray<int, (block_size_y * block_size_x), bsg_tiles_X, bsg_tiles_Y, 1>;





void __attribute__ ((noinline)) subblock2shmem (int *A, Array_A sh_dest, int M, int N, int sub_block_y, int sub_block_x) { 

	const int block_y = block_size_y;
	const int block_x = BLOCK_WIDTH;

	int start_y = sub_block_y * block_y;
	int start_x = sub_block_x * block_x;
	
	for (int iter_y = __bsg_y; iter_y < block_y; iter_y += bsg_tiles_Y) { 
		for (int iter_x = __bsg_x; iter_x < block_x; iter_x += bsg_tiles_X) { 
			// sh_dest[iter_y][iter_x] <-- A[iter_y + start_y][iter_x + start_x]
			//bsg_tile_group_shared_store (int, sh_dest, (iter_y * block_size_x + iter_x), A[((iter_y + start_y) * N + iter_x + start_x)]);
			sh_dest[iter_y * block_x + iter_x] = A[((iter_y + start_y) * N + iter_x + start_x)]; 
		}
	}
	return; 
}


void __attribute__ ((noinline)) subblock2shmem_xposed (int *A, Array_B sh_dest, int M, int N, int sub_block_y, int sub_block_x) { 

	const int block_y = BLOCK_WIDTH;
	const int block_x = block_size_x;

	int start_y = sub_block_y * block_y;
	int start_x = sub_block_x * block_x;
	
	for (int iter_y = __bsg_y; iter_y < block_y; iter_y += bsg_tiles_Y) { 
		for (int iter_x = __bsg_x; iter_x < block_x; iter_x += bsg_tiles_X) { 
			// sh_dest[iter_x][iter_y] <-- A[iter_y + start_y][iter_x + start_x]
			//bsg_tile_group_shared_store (int, sh_dest, (iter_x * block_size_y + iter_y), A[((iter_y + start_y) * N + iter_x + start_x)]);
			sh_dest[iter_x * block_y + iter_y] = A[((iter_y + start_y) * N + iter_x + start_x)];
		}
	}
	return; 
}


void __attribute__ ((noinline)) shmem2subblock (int *A, Array_C sh_src, int M, int N, int sub_block_y, int sub_block_x) { 

	const int block_y = block_size_y;
	const int block_x = block_size_x;

	int start_y = sub_block_y * block_y;
	int start_x = sub_block_x * block_x;
	
	for (int iter_y = __bsg_y; iter_y < block_y; iter_y += bsg_tiles_Y) { 
		for (int iter_x = __bsg_x; iter_x < block_x; iter_x += bsg_tiles_X) { 
			// A[iter_y + start_y][iter_x + start_x] <-- sh_src[iter_y][iter_x]
			//bsg_tile_group_shared_load (int, sh_src, (iter_y * block_size_x + iter_x), A[((iter_y + start_y) * N + iter_x + start_x)]);
			A[((iter_y + start_y) * N + iter_x + start_x)] = sh_src[iter_y * block_x + iter_x];
		}
	}
	return; 
}


void __attribute__ ((noinline)) subblock_shmem_matrix_mul_xposed (Array_A sh_A, Array_B sh_B, Array_C sh_C, int M, int N, int P, int block_num) { 

	
	for (int iter_y = __bsg_y; iter_y < block_size_y; iter_y += bsg_tiles_Y) { 
		for (int iter_x = __bsg_x; iter_x < block_size_x; iter_x += bsg_tiles_X) { 

			int sum = 0; 
			int lc_A, lc_B;
			for (int k = 0; k < BLOCK_WIDTH; k ++) { 
				// lc_A <-- sh_A[iter_y][iter_x]
				//bsg_tile_group_shared_load (int, sh_A, (iter_y * BLOCK_WIDTH + k), lc_A); 
				lc_A = sh_A[iter_y * BLOCK_WIDTH + k];
				// lc_B <-- sh_B[iter_y][iter_x]	remember B is transposed
				// bsg_tile_group_shared_load (int, sh_B, (iter_x * BLOCK_WIDTH + k), lc_B);
				lc_B = sh_B[iter_x * BLOCK_WIDTH + k];
				sum += lc_A * lc_B;
			}

			if (!block_num) { 
				// sh_C[iter_y][iter_x] <-- sum
				// bsg_tile_group_shared_store (int, sh_C, (iter_y * block_size_x + iter_x), sum);
				sh_C[iter_y * block_size_x + iter_x] = sum;
			}
			else { 
				int lc_C;
				// sh_C[iter_y][iter_x] += sum
				// bsg_tile_group_shared_load (int, sh_C, (iter_y * block_size_x + iter_x), lc_C);
				lc_C = sh_C[iter_y * block_size_x + iter_x];
				// bsg_tile_group_shared_store (int, sh_C, (iter_y * block_size_x + iter_x), lc_C + sum);
				sh_C[iter_y * block_size_x + iter_x] = lc_C + sum;
			} 
		}
	}
	return;
}





using Array = TileGroupStripedArray<struct foo, 16, bsg_tiles_X, bsg_tiles_Y, 2>;


extern "C" int __attribute__ ((noinline)) kernel_matrix_mul_striped_shared_mem(int *A, int *B, int *C, int M, int N, int P) {

	
	// declare tile-group shared memory
//	bsg_tile_group_shared_mem (int, sh_A, (block_size_y * BLOCK_WIDTH));
//	bsg_tile_group_shared_mem (int, sh_B, (BLOCK_WIDTH * block_size_x));
//	bsg_tile_group_shared_mem (int, sh_C, (block_size_y * block_size_x));


	Array_A sh_A;
	Array_B sh_B;
	Array_C sh_C;


	int num_blocks = N / BLOCK_WIDTH;	// *** Must divide evenly

	for (int block_num = 0; block_num < num_blocks; block_num ++) { 

		subblock2shmem (       A, sh_A, M, N, __bsg_tile_group_id_y, block_num);
 
		subblock2shmem_xposed (B, sh_B, N, P, block_num, __bsg_tile_group_id_x);

		bsg_tile_group_barrier (&r_barrier, &c_barrier);
		
		subblock_shmem_matrix_mul_xposed (sh_A, sh_B, sh_C, M, N, P, block_num);
		
		bsg_tile_group_barrier (&r_barrier, &c_barrier); 
	}

	shmem2subblock (C, sh_C, M, P, __bsg_tile_group_id_y, __bsg_tile_group_id_x); 

	bsg_tile_group_barrier (&r_barrier, &c_barrier) ;

	return 0;
}
