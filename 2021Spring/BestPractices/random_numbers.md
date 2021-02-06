# Random Number Generation in C++

In scientific computing, it often becomes necessary to produce sets of (pseudo-) random numbers. As with many tools which have been updated in C++, the best practice for new programs is to eschew the C-style `rand` function in favor of the new C++ `<random>` library.

For some background on Pseudo-Random Number Generators (PRNGs), read the next section. If you don't care about the background and just want to know how why `rand` isn't recommended, [continue below](#whats-wrong-with-rand).

## Random Number Generators

Computers are largely deterministic in their function. A sane program which is given the same set of inputs will produce the same outputs every time that it is run. This is almost always a _Good Thing_, but it makes the generation of random numbers somewhat difficult. After all, how does a deterministic system produce random results? There are a few ways in which this can be done depending on the needs of the application. 

For cryptography, random numbers need to be "truly" random, at least in the sense that they are unpredictable to an observer. Some CPUs provide sources of _hardware randomness_ which rely on stochastic processes within the CPU in order to generate unpredictable bits. Other systems implement _entropy pools_ which combine physical phenomena in the form of user input, network latency, and other processes to create a pool of random bits.

In scientific domains, however, reproducibility is a vital part of research. Results must be reproducible right down to their randomized components in order to be validated. In this case, applications often rely on Pseudo-random Number Generators, or **PRNG**s. These generators are algorithms which produce random bits based on some complex mathematical operation. This function is "seeded" with some starting value and then called repeatedly to generate new values with each iteration.

## What's wrong with `rand`?

The `rand` function in C (or `std::rand` in C++) has long been plagued by two major problems.

### The typical usage does not generate a uniform distribution

Consider the following idiom:
```c++
const int RANGE = 10;
int random_value = rand() % (RANGE + 1);
```

This idiom returns pseudo-random positive integers between `0` and `RANGE`, but the values aren't uniformly distributed. Depending on the range and the seed, this poor distribution can even bias the results of an experiment.

### Some implementations of the PRNG function produce patterns in the lower-order bits

If it is not restricted to a given range, `rand` returns integral values between `0` and the implementation-defined constant `RAND_MAX`. In some implementations, the least significant bits of these integers will begin to show patterns after `RAND_MAX` values have been returned. A particularly heinous pattern is that of the lowest-order bit always being the same every `RAND_MAX` iterations. This means that if the `n`th call to `rand` returns an even number, the `n + RAND_MAX`th call will also return an even number. It becomes even more serious implementations with a very small `RAND_MAX` (Microsoft Windows, for example, defines `RAND_MAX` as `32767`).

These issues can be remedied somewhat by clever manipulation of the output of `rand()`, but there's only so much that can be done with a low-quality algorithm.

## The C++ Way

(coming soon)



