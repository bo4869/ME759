#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>

// class holds on to statistics related to the scheduling process
class schedulingStatistics {
public:
    std::atomic<unsigned int> nTimesConsumerHeldBack;
    std::atomic<unsigned int> nTimesProducerHeldBack;
    std::atomic<unsigned int> nConsumerUpdates;
    std::atomic<unsigned int> nProducerUpdates;

    schedulingStatistics() noexcept {
        nTimesConsumerHeldBack = 0;
        nTimesProducerHeldBack = 0;
        nConsumerUpdates = 0;
        nProducerUpdates = 0;
    }

    ~schedulingStatistics() {}
};

// class that will be used via an atomic object to coordinate the production-consumption interplay
class schedulingSupport {
public:
    std::atomic<int> stampLastUpdateOfConsumer;
    std::atomic<int> currentStampOfConsumer;
    std::atomic<int> consumerRequestedUpdateFrequency;
    std::atomic<bool> consumerDone;

    std::atomic<bool> consOwned_Prod2ConsBuffer_isFresh;
    std::atomic<bool> prodOwned_Cons2ProdBuffer_isFresh;

    std::mutex consOwnedBuffer_AccessCoordination;
    std::mutex prodOwnedBuffer_AccessCoordination;
    std::mutex producerCanProceed;
    std::mutex consumerCanProceed;
    std::condition_variable cv_ProducerCanProceed;
    std::condition_variable cv_ConsumerCanProceed;
    schedulingStatistics schedulingStats;


    schedulingSupport() noexcept {
        consumerRequestedUpdateFrequency = -1; // that is, let consumer advance into future as much as it wants
        stampLastUpdateOfConsumer = -1;
        currentStampOfConsumer = 0;
        consumerDone = false;
        consOwned_Prod2ConsBuffer_isFresh = false;
        prodOwned_Cons2ProdBuffer_isFresh = false;
    }

    ~schedulingSupport() {}

    inline bool consumerShouldWait()const {
        // do not hold consumer back under the following circustances:
        // * the update frequency is negative, consumer can drift into future
        // * the producer is done
        if (consumerRequestedUpdateFrequency < 0) return false;

        // the consumer should wait if it moved too far into the future
        bool shouldWait = (currentStampOfConsumer > stampLastUpdateOfConsumer + consumerRequestedUpdateFrequency ? true : false);
        return shouldWait;
    }
};



