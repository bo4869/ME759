#pragma once
#include <mutex>
#include "problemParams.h"

class schedulingSupport;

class cConsumer {
private:
    schedulingSupport* pSchedSupport;
    int* pProducerOwned_TransfBuffer; // pointer to remote buffer where producer stores work-order data provided by the consumer
    int transferBuffer[N_MANUFACTURED_ITEMS] = { 0, 0, 0 };
    int outcome[N_MANUFACTURED_ITEMS];
    int input4Producer[N_INPUT_ITEMS] = { -1,-2,-3,-4 };
    
    int consAverageTime; // time required in the consumption process; fake lag
    int nConsumerCycles;

    int localUse(int val);

public:
    cConsumer(schedulingSupport* pSchedSup) :pSchedSupport(pSchedSup) {
        pProducerOwned_TransfBuffer = NULL;
        nConsumerCycles=0; 
        consAverageTime = 0;
    }
    ~cConsumer() {}

    void setConsumerAverageTime(int val) { consAverageTime = val; }
    void setDestinationBuffer(int* pPB) { pProducerOwned_TransfBuffer = pPB; }
    void setNConsumerCycles(int val) { nConsumerCycles = val; }
    int* pDestinationBuffer() { return transferBuffer; }
    void operator() ();
};