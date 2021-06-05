// create this file indepented of cycles and time, maybe already print from here,
// depending on the measurement type

// inline functions for speed
// https://gcc.gnu.org/onlinedocs/gcc/Inline.html

#include "benchmark.h"

Benchmark::Benchmark()
{
	init();
}

Benchmark::~Benchmark()
{
	clear();
}


#ifndef CYCLES

	void Benchmark::init()
	{	
		running = 0;
		begin = 0;
		end = 0;
		timer.reset();
	}

	void Benchmark::start()
	{
		running = 1;
		timer.start();
		begin = timer.read_us();
	}


	void Benchmark::stop()
	{
		running = 0;
		timer.stop();
		end = timer.read_us();
	}

	uint32_t Benchmark::read()
	{
		return uint32_t(end - begin);
	}

	void Benchmark::clear()
	{
		timer.reset();
	}

#endif // NOT CYCLES



#ifdef CYCLES

	void Benchmark::init()
	{
		cycles = 0;
		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CYCCNT = 0;
	}

	void Benchmark::start()
	{
		running = 1;
		DWT->CTRL = 0x40000001;
	}


	void Benchmark::stop()
	{
		running = 0;
		DWT->CTRL = 0x40000000;
	}

	uint32_t Benchmark::read()
	{
		cycles = DWT->CYCCNT;
		return cycles;
	}

	void Benchmark::clear()
	{
		cycles = 0;
	  	DWT->CYCCNT = 0;
	}

#endif // CYCLES