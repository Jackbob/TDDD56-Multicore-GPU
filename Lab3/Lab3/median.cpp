/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>

#include "support.h"


unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	// your code here
    int n_pxvalues[256] = {0};
    int totalpx = ox * oy;
    int current_pxcount = 0;
    

    for (int y = -oy; y <= oy; ++y)
		for (int x = -ox; x <= ox; x += elemPerPx)
			++n_pxvalues[image[y*(int)stride+x]];

    int prevpx = 0;
    for(int i=0; i<256; ++i){
        current_pxcount += n_pxvalues[i];
        if(current_pxcount > totalpx/2)
            return (prevpx != 0) ? (i+prevpx)/2 : i;

        prevpx = (n_pxvalues[i] != 0) ? i : prevpx;
            
    }

    return 0;
}



int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << "input output radius [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[4])};
	
	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";
		
	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
	
	// Skeleton instance
	auto calculateMedian = skepu2::MapOverlap(median_kernel);
	calculateMedian.setBackend(spec);
	calculateMedian.setOverlap(radius, radius*imageInfo.elementsPerPixel);
	
	auto timeTaken = skepu2::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	
	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";
	
	return 0;
}


