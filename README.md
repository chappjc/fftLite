# fftLite
**Work in progress!**

The fftLite class is a very simple ("lite") FFT implementation.

This is mainly an effort to learn how the FFT can be computed (not just a naive DFT). Another goals is to produce a *very small* implementation that can be used where performance is not critical and without worrying about code bloat or large third-party libraries.  It's not terribly fast, especially for non-power-of-two lengths!  There is much that still needs to be implemented, such as different radix sizes, but it's a start and the output matches MATLAB's `fft` (close to machine precision).
