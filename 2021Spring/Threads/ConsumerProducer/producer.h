#pragma once
#include <mutex>
#include "problemParams.h"

class schedulingSupport;

class cProducer {
private:
    schedulingSupport * pSchedSupport;
    int* pConsumerOwned_TransfBuffer; // this is where the consumer stores data that needs to be produced herein
    int product[N_MANUFACTURED_ITEMS] = { 1,2,3 };
    int transferBuffer[N_INPUT_ITEMS] = { 0, 0, 0, 0 };
    int inputData[N_INPUT_ITEMS];
    int prodAverageTime;
    int costlyProductionStep(int) const;

public:
    cProducer(schedulingSupport* pSchedSup) :pSchedSupport(pSchedSup){
        prodAverageTime = 0;
        pConsumerOwned_TransfBuffer = NULL;
    }
    ~cProducer() {}

    void setProducerAverageTime(int val) { prodAverageTime = val; }
    void setDestinationBuffer(int* pCB) { pConsumerOwned_TransfBuffer = pCB; }
    int* pDestinationBuffer() { return transferBuffer; }
    void primeConsumer();
    void operator() ();
};