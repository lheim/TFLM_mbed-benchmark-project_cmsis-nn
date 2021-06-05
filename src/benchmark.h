#include <mbed.h>


class Benchmark {

public:
	Benchmark();
	~Benchmark();
	void init();
	void start();
	void stop();
	uint32_t read();
	void clear();

protected:
	int running;
	int begin;
	int end;
	uint32_t cycles;
	Timer timer;
};


