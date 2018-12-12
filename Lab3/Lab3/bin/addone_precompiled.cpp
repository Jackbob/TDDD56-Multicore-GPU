#ifndef SKEPU_PRECOMPILED
#define SKEPU_PRECOMPILED
#endif
#ifndef SKEPU_OPENMP
#define SKEPU_OPENMP
#endif
#ifndef SKEPU_OPENCL
#define SKEPU_OPENCL
#endif
/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <iostream>

#include <skepu2.hpp>

float addOneFunc(float a)
{
	return a+1;
}
struct skepu2_userfunction_addOneMap_addOneFunc
{
constexpr static size_t totalArity = 1;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<float>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<>;
constexpr static skepu2::AccessMode anyAccessMode[] = {
};

using Ret = float;

constexpr static bool prefersMatrix = 0;

#define SKEPU_USING_BACKEND_OMP 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block)
#define VARIANT_OPENMP(block) block
#define VARIANT_CUDA(block)
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a)
{
	return a+1;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a)
{
	return a+1;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_addOneMap_addOneFunc::anyAccessMode[];




#include "addone_precompiled_MapKernel_addOneFunc_arity_1_cl_source.inl"
int main(int argc, const char* argv[])
{
	/* Program parameters */
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}
	
	const size_t size = std::stoul(argv[1]);
	
	/* Skeleton instances */
	skepu2::backend::Map<1, skepu2_userfunction_addOneMap_addOneFunc, bool, CLWrapperClass_addone_precompiled_MapKernel_addOneFunc_arity_1> addOneMap(false);
	
	/* SkePU containers */
	skepu2::Vector<float> input(size), res(size);
	input.randomize(0, 9);
	
	
	// This is how to measure execution times with SkePU
	auto dur = skepu2::benchmark::measureExecTime([&]
	{
		// Code to be measured here
		addOneMap(res, input);
	});
	
	/* This is how to print the time */
	std::cout << "Time: " << (dur.count() / 10E6) << " seconds.\n";
	
	
	/* Print vector for debugging */
//	std::cout << "Input:  " << input << "\n";
//	std::cout << "Result: " << res << "\n";
	
	
	return 0;
}

