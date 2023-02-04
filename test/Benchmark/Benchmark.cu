#define NR_BITS 16
#define NR_CHANNELS 480
#define NR_POLARIZATIONS 2
#define NR_SAMPLES_PER_CHANNEL 3072
#define NR_RECEIVERS 576
#define NR_BASELINES ((NR_RECEIVERS) * ((NR_RECEIVERS) + 1) / 2)
#define NR_RECEIVERS_PER_BLOCK 64
#define NR_TIMES_PER_BLOCK (128 / (NR_BITS))

#include "libtcc/Correlator.h"


#include <iostream>
#include <complex>

#include <cuda.h>
#include <cuda_fp16.h>

inline void checkCudaCall(cudaError_t error) 
{
	if (error != cudaSuccess) { 
		std::cerr << "error " << error << std::endl;
		exit(1);
	}
}

typedef std::complex<__half>	Sample;
typedef std::complex<float>		Visibility;

typedef Sample Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_RECEIVERS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

int main(int argc, char* argv[]) {	
	std::cout << "Benchmarking in progress..." << std::endl;
	
	try {
		checkCudaCall(cudaSetDevice(0)); // combine the CUDA rutime API and CUDA driver API
		checkCudaCall(cudaFree(0));
		
		// create correlator 
		tcc::Correlator correlator(NR_BITS, NR_RECEIVERS, NR_CHANNELS, NR_SAMPLES_PER_CHANNEL, NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);
		
		cudaStream_t stream;
		Samples *samples;
		Visibilities *visibilities;
		
		// create stream and allocate memory 
		checkCudaCall(cudaStreamCreate(&stream));
		checkCudaCall(cudaMallocManaged(&samples, sizeof(Samples)));
		checkCudaCall(cudaMallocManaged(&visibilities, sizeof(Visibilities)));
		
		// initialise 2 values at the input as (2+3i) and (4+5i)
		(*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][174][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(2, 3);
		(*samples)[NR_CHANNELS / 3][NR_SAMPLES_PER_CHANNEL / 5 / NR_TIMES_PER_BLOCK][418][0][NR_SAMPLES_PER_CHANNEL / 5 % NR_TIMES_PER_BLOCK] = Sample(4, 5);
	
		// run correlation operation
		correlator.launchAsync((CUstream) stream, (CUdeviceptr) visibilities, (CUdeviceptr) samples);
		checkCudaCall(cudaDeviceSynchronize());
		
		// at this particular output cell (2+3i)(4-5i) = (23-2i)
		// conjugated as we are correlating
		std::cout << ((*visibilities)[160][87745][0][0] == Visibility(23, 2) ? "success" : "failed") << std:: endl;
		
		checkCudaCall(cudaFree(visibilities));
		checkCudaCall(cudaFree(samples));
		checkCudaCall(cudaStreamDestroy(stream));
	} catch (std::exception &error) {
		std::cerr << error.what() << std::endl;
	}
	return 0;
}
