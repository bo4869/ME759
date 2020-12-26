#include <iostream>
#include <thread>
#include <sstream>

#include "schedulingSupport.h"
#include "producer.h"
#include "consumer.h"

void processCommandLineInput(int argc, char* argv[], int& updateFreq, int& consTime, int& prodTime, int& nConCycl) {
    // Check the number of parameters
    if (argc < 5) {
        std::cerr << "Usage of " << argv[0] << " requires several input params." << std::endl;
        std::cerr << "Update Frequency" << std::endl;
        std::cerr << "Consumer time" << std::endl;
        std::cerr << "Producer time" << std::endl;
        std::cerr << "Number of consumer cycles" << std::endl;
        exit(-1);
    }
    std::istringstream iss;
    iss = std::istringstream(argv[1]); iss >> updateFreq;
    iss = std::istringstream(argv[2]); iss >> consTime;
    iss = std::istringstream(argv[3]); iss >> prodTime;
    iss = std::istringstream(argv[4]); iss >> nConCycl;
}

int main(int argc, char* argv[])
{
    // get input data from command line
    int updateFreq;
    int timeConsumerSide;
    int timeProducerSide;
    int nConsumerCycles;
    processCommandLineInput(argc, argv, updateFreq, timeConsumerSide, timeProducerSide, nConsumerCycles);

    schedulingSupport schedSupport;
    schedSupport.consumerRequestedUpdateFrequency = updateFreq;

    cConsumer consGuy(&schedSupport);
    cProducer prodGuy(&schedSupport);

    // set up the producer
    int* pBuffer = consGuy.pDestinationBuffer();
    prodGuy.setDestinationBuffer(pBuffer);
    prodGuy.primeConsumer();
    prodGuy.setProducerAverageTime(timeProducerSide);

    // set up the consumer
    pBuffer = prodGuy.pDestinationBuffer();
    consGuy.setDestinationBuffer(pBuffer);
    consGuy.setConsumerAverageTime(timeConsumerSide);
    consGuy.setNConsumerCycles(nConsumerCycles);

    // get the threads going
    std::thread prodThread(std::ref(prodGuy));
    std::thread consThread(std::ref(consGuy));

    consThread.join();
    prodThread.join();

    // Sim statistics
    std::cout << "\n~~ SIM STATISTICS ~~\n";
    std::cout << "Number of consumer updates: " << schedSupport.schedulingStats.nConsumerUpdates << std::endl;
    std::cout << "Number of producer updates: " << schedSupport.schedulingStats.nProducerUpdates << std::endl;
    std::cout << "Number of times consumer held back: " << schedSupport.schedulingStats.nTimesConsumerHeldBack << std::endl;
    std::cout << "Number of times producer held back: " << schedSupport.schedulingStats.nTimesProducerHeldBack << std::endl;

    return 0;
}