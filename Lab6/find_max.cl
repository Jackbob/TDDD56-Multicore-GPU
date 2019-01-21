__kernel void find_max(__global unsigned int *data, const unsigned int length, __local unsigned int *sharedData)
{ 
  unsigned int tid = get_local_id(0);
  unsigned int i = get_global_id(0);
    
  if (i < length) sharedData[tid] = data[i];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int index = get_local_size(0) / 2; index > 0; index >>= 1 ) {
  	if (tid < index && sharedData[tid + index] > sharedData[tid]) {
  		sharedData[tid] = sharedData[tid + index];
  	}
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0) {
    data[get_group_id(0)] = sharedData[0];
  }
  
}
