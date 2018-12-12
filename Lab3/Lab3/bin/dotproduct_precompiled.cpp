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

/* SkePU user functions */


float dot_func(float element1, float element2)
{
	return element1 * element2;
}
struct skepu2_userfunction_dot_multstep_dot_func
{
constexpr static size_t totalArity = 2;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<float, float>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float element1, float element2)
{
	return element1 * element2;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float element1, float element2)
{
	return element1 * element2;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_dot_multstep_dot_func::anyAccessMode[];

struct skepu2_userfunction_dot_skepu_dot_func
{
constexpr static size_t totalArity = 2;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<float, float>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float element1, float element2)
{
	return element1 * element2;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float element1, float element2)
{
	return element1 * element2;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_dot_skepu_dot_func::anyAccessMode[];


// more user functions...
float plusfloat(float a, float b){

	return a + b;
}
struct skepu2_userfunction_dot_reducestep_plusfloat
{
constexpr static size_t totalArity = 2;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<>;
using ContainerArgs = std::tuple<>;
using UniformArgs = std::tuple<float, float>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{

	return a + b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{

	return a + b;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_dot_reducestep_plusfloat::anyAccessMode[];

struct skepu2_userfunction_dot_skepu_plusfloat
{
constexpr static size_t totalArity = 2;
constexpr static bool indexed = 0;
using ElwiseArgs = std::tuple<float, float>;
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
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float OMP(float a, float b)
{

	return a + b;
}
#undef SKEPU_USING_BACKEND_OMP

#define SKEPU_USING_BACKEND_CPU 1
#undef VARIANT_CPU
#undef VARIANT_OPENMP
#undef VARIANT_CUDA
#define VARIANT_CPU(block) block
#define VARIANT_OPENMP(block)
#define VARIANT_CUDA(block) block
static inline SKEPU_ATTRIBUTE_FORCE_INLINE float CPU(float a, float b)
{

	return a + b;
}
#undef SKEPU_USING_BACKEND_CPU
};

constexpr skepu2::AccessMode skepu2_userfunction_dot_skepu_plusfloat::anyAccessMode[];





#include "dotproduct_precompiled_ReduceKernel_plusfloat_cl_source.inl"

#include "dotproduct_precompiled_MapKernel_dot_func_arity_2_cl_source.inl"

#include "dotproduct_precompiled_MapReduceKernel_dot_func_plusfloat_arity_2_cl_source.inl"
int main(int argc, const char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <input size> <backend>\n";
		exit(1);
	}

	const size_t size = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
//	spec.setCPUThreads(<integer value>);


	/* Skeleton instances */
	skepu2::backend::MapReduce<2, skepu2_userfunction_dot_skepu_dot_func, skepu2_userfunction_dot_skepu_plusfloat, bool, bool, CLWrapperClass_dotproduct_precompiled_MapReduceKernel_dot_func_plusfloat_arity_2> dot_skepu(false, false);
	skepu2::backend::Map<2, skepu2_userfunction_dot_multstep_dot_func, bool, CLWrapperClass_dotproduct_precompiled_MapKernel_dot_func_arity_2> dot_multstep(false);
	skepu2::backend::Reduce1D<skepu2_userfunction_dot_reducestep_plusfloat, bool, CLWrapperClass_dotproduct_precompiled_ReduceKernel_plusfloat> dot_reducestep(false);

// ...

	/* Set backend (important, do for all instances!) */
//	instance.setBackend(spec);

	/* SkePU containers */
	skepu2::Vector<float> result(size, 0.0f), v1(size, 1.0f), v2(size, 2.0f);




	/* Compute and measure time */
	float resComb, resSep;

	auto timeComb = skepu2::benchmark::measureExecTime([&]
	{
		resComb = dot_skepu(v1,v2);
	});

	auto timeSep = skepu2::benchmark::measureExecTime([&]
	{
		dot_multstep(result, v1, v2);
		resSep = dot_reducestep(result);
	});

	std::cout << "Time Combined: " << (timeComb.count() / 10E6) << " seconds.\n";
	std::cout << "Time Separate: " << ( timeSep.count() / 10E6) << " seconds.\n";


	std::cout << "Result Combined: " << resComb << "\n";
	std::cout << "Result Separate: " << resSep  << "\n";

	return 0;
}
