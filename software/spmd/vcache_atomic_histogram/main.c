#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_manycore_atomic.h"

#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include "bsg_tile_group_barrier.h"

#define N 1024

int lock __attribute__ ((section (".dram"))) = {0};
int histogram[32] __attribute__ ((section (".dram"))) = {0};
int data[N] __attribute__ ((section (".dram"))) =
{
27,24,13,8,16,12,25,9,15,18,29,16,9,24,19,8,29,31,25,28,9,23,28,21,15,3,13,19,29,30,15,27,
8,25,17,0,23,12,26,21,0,15,27,7,10,27,6,18,7,30,25,14,2,10,16,29,3,17,22,17,26,17,30,19,
18,14,19,12,18,9,6,5,19,21,15,2,24,28,29,26,28,29,17,12,22,8,25,27,28,18,30,18,14,21,31,29,
25,2,19,15,20,27,7,23,3,7,25,10,26,3,4,22,1,18,29,17,21,0,20,19,18,12,11,31,1,0,30,5,
3,6,25,29,0,13,3,8,7,20,11,5,16,1,3,31,6,11,23,26,29,5,21,30,1,21,27,10,8,19,14,5,
15,13,18,16,9,11,26,8,17,0,23,10,1,8,7,30,11,9,11,30,20,19,22,12,13,20,0,6,10,7,20,12,
28,18,13,12,22,13,21,1,14,8,5,16,15,17,24,28,15,9,14,25,28,25,6,31,20,2,23,31,12,21,10,6,
22,0,26,16,3,3,20,27,8,31,3,27,12,2,8,14,25,27,4,16,20,11,27,8,0,1,21,17,30,30,29,1,
23,22,20,22,28,20,11,17,6,18,0,4,10,25,22,10,19,1,5,31,9,12,17,9,15,7,1,5,16,2,12,10,
13,3,29,15,26,31,10,15,22,13,9,23,28,29,20,12,31,20,2,2,23,1,0,12,16,14,15,18,21,13,11,31,
8,24,13,11,2,27,22,28,14,21,3,12,6,1,30,6,4,6,12,17,4,31,31,4,12,21,28,15,29,10,15,15,
21,6,19,7,10,30,28,26,1,4,8,25,26,18,22,25,2,2,27,1,7,1,0,27,10,5,4,20,30,16,28,16,
18,21,25,24,31,23,28,6,17,19,26,15,25,12,18,27,25,21,0,5,16,8,2,27,30,9,13,25,1,20,4,9,
26,1,1,13,15,27,22,21,4,31,13,19,12,1,15,4,1,19,20,3,17,11,12,24,15,28,19,14,20,10,3,21,
19,25,4,29,25,29,27,21,25,16,25,6,25,14,24,14,25,2,1,29,15,28,30,21,18,6,2,26,28,24,22,13,
9,3,13,18,29,29,13,3,24,23,0,14,21,0,29,30,23,2,2,11,0,11,0,31,26,2,28,6,6,21,30,3,
0,11,0,19,27,5,3,11,30,4,30,11,15,9,29,30,20,5,31,3,18,5,28,30,25,10,7,24,9,13,1,4,
0,2,2,13,17,23,4,13,20,2,14,11,30,1,13,13,23,10,6,9,15,30,25,8,17,22,25,14,12,24,13,7,
14,29,4,14,20,15,6,0,22,19,0,9,24,20,17,4,22,15,21,24,7,24,8,31,3,28,1,8,16,18,12,3,
8,9,24,29,19,1,25,9,10,16,7,29,5,13,9,16,18,20,17,13,20,12,24,25,9,11,20,5,22,12,18,4,
21,11,15,13,15,22,10,20,1,9,23,1,19,0,15,28,0,16,2,27,21,23,21,0,1,19,31,27,22,23,7,24,
9,3,14,10,5,13,28,13,14,22,16,4,29,14,25,12,25,12,7,6,30,18,1,12,7,2,5,1,20,5,19,19,
22,16,9,28,11,14,20,16,30,30,29,29,18,15,22,6,8,1,5,0,20,4,25,21,31,12,29,14,10,3,28,25,
10,14,10,0,1,11,6,16,6,6,21,23,9,27,8,11,22,1,29,2,14,23,1,25,31,14,3,2,3,24,13,29,
14,2,13,24,26,1,5,15,4,27,29,10,13,17,9,17,6,9,14,19,17,8,7,3,25,3,23,7,9,23,21,23,
16,27,3,20,3,23,11,21,22,21,7,26,7,16,21,7,20,9,5,25,17,10,18,0,4,12,31,16,2,24,25,24,
18,22,6,23,26,24,11,18,20,28,3,26,16,11,14,0,7,20,21,15,30,15,10,27,8,19,22,26,25,12,1,1,
23,30,10,14,23,21,8,21,9,11,17,23,4,0,20,0,1,7,20,2,1,31,13,28,6,13,11,5,10,31,23,12,
13,8,17,23,21,14,1,29,13,12,0,4,27,16,23,4,10,26,26,7,0,25,5,25,21,5,2,29,19,19,14,4,
19,8,25,23,0,29,1,2,9,4,7,11,23,12,8,15,12,9,28,17,31,24,18,8,21,14,23,12,15,0,23,1,
21,18,24,9,21,6,16,10,31,31,6,18,10,30,29,18,23,21,11,29,28,10,23,0,26,18,30,11,20,10,25,19,
31,0,4,1,4,29,30,15,30,26,24,23,6,17,13,30,5,5,21,5,3,16,25,19,24,8,9,13,31,22,30,17
};

INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, bsg_tiles_X-1, 0, bsg_tiles_Y-1);

void do_histogram_work()
{
  int local_histogram[32] = {0};

  int work = N/(bsg_tiles_X * bsg_tiles_Y);
  int start_idx = __bsg_id * work;
  bsg_printf("%d\n", start_idx);
  for (int i = 0; i < work; i++)
  {
    int local_data = data[start_idx+i];
    local_histogram[local_data]++;
  }

  // get the lock
  int lock_val = 1;

  do {
    lock_val = bsg_amoswap_aq(&lock, 1);
  } while (lock_val != 0); 

  // critical region
  //bsg_printf("updating histogram. x=%d y=%d\n", __bsg_x, __bsg_y);
  for (int i = 0; i < 32; i++)
  {
    int local_hist = histogram[i];
    histogram[i] = local_hist + local_histogram[i];
  }

  // release
  bsg_amoswap_rl(&lock, 0);
}

void validate()
{
  if (histogram[0] != 37) bsg_fail();
  if (histogram[1] != 43) bsg_fail();
  if (histogram[2] != 29) bsg_fail();
  if (histogram[3] != 31) bsg_fail();
  if (histogram[4] != 29) bsg_fail();
  if (histogram[5] != 27) bsg_fail();
  if (histogram[6] != 30) bsg_fail();
  if (histogram[7] != 26) bsg_fail();
  if (histogram[8] != 29) bsg_fail();
  if (histogram[9] != 34) bsg_fail();
  if (histogram[10] != 33) bsg_fail();
  if (histogram[11] != 30) bsg_fail();
  if (histogram[12] != 35) bsg_fail();
  if (histogram[13] != 37) bsg_fail();
  if (histogram[14] != 32) bsg_fail();
  if (histogram[15] != 35) bsg_fail();
  if (histogram[16] != 29) bsg_fail();
  if (histogram[17] != 27) bsg_fail();
  if (histogram[18] != 31) bsg_fail();
  if (histogram[19] != 30) bsg_fail();
  if (histogram[20] != 35) bsg_fail();
  if (histogram[21] != 42) bsg_fail();
  if (histogram[22] != 28) bsg_fail();
  if (histogram[23] != 37) bsg_fail();
  if (histogram[24] != 27) bsg_fail();
  if (histogram[25] != 43) bsg_fail();
  if (histogram[26] != 23) bsg_fail();
  if (histogram[27] != 26) bsg_fail();
  if (histogram[28] != 29) bsg_fail();
  if (histogram[29] != 37) bsg_fail();
  if (histogram[30] != 34) bsg_fail();
  if (histogram[31] != 29) bsg_fail();

  bsg_finish();
}

int main()
{

  bsg_set_tile_x_y();
  /*
  if (__bsg_id == 0)
  {
    bsg_printf("%d\n", bsg_tiles_X);
    bsg_printf("%d\n", bsg_tiles_Y);
  }
  */

  do_histogram_work();

  bsg_tile_group_barrier(&r_barrier, &c_barrier);  

  if (__bsg_id == 0) validate();



  bsg_wait_while(1);
}

