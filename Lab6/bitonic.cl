__kernel void bitonic(__global unsigned int *data, const unsigned int outer_length, const unsigned int inner_length)
{ 
  unsigned int tid = get_local_id(0);
  unsigned int i = get_global_id(0);
  
  //for (unsigned int inner_length = outer_length / 2; inner_length > 0; inner_length >>= 1 ) {
    //barrier(CLK_LOCAL_MEM_FENCE);

    int ixj = i ^ inner_length;
    if ((ixj) > i)
    {
      if ((i & outer_length)==0 && data[i]>data[ixj]) {
        unsigned int a = data[i];
        data[i] = data[ixj];
        data[ixj] = a;
      }
      if ((i & outer_length)!=0 && data[i]<data[ixj]) {
        unsigned int a = data[i];
        data[i] = data[ixj];
        data[ixj] = a;
      }
    }
 // }
  
}
