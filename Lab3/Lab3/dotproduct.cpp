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

// more user functions...
float plusfloat(float a, float b){

	return a + b;
}



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
	auto dot_skepu = skepu2::MapReduce<2>(dot_func, plusfloat);
	auto dot_multstep = skepu2::Map<2>(dot_func);
	auto dot_reducestep = skepu2::Reduce<>(plusfloat);

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
