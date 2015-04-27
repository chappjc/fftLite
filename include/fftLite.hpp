// fftLite.hpp - The fftLite class is a very simple ("lite") FFT implementation.
//
// Instantiating the class with the desired FFT length (N) does the following:
//   1. Allocates space for temporary arrays.
//   2. Computes cos/sin lookup tables (twiddle factors).  These are the 
//      root-of-unity complex coefficients use in the butterfly operations of 
//      the Cooley-Tukey FFT algorithm.
//   3. If using Bluestein's algorithm for non-power-of-two length, the complex
//      chirp is also pre-computed.
//
// by Jonathan Chappelow (chappjc)
// Inspired by descriptions and code at Project Nayuki

class fftLite
{
public:
    fftLite(size_t N_);
    fftLite();
    ~fftLite();

    int fft(double *realIO, double *imagIO);
    int ifft(double *realIO, double *imagIO);

    int fft(double const *realIN, double *real, double *imag);

private:

    int N;
    int log2N;
    int M;

    double *cosLUT;
    double *sinLUT;
    int *revBits;
    double *cosLUT_blu;
    double *sinLUT_blu;

    double *temps;

private:

    static unsigned int reverseBits(unsigned int x, unsigned int n);

    int bluestein(double *realIO, double *imagIO) const;
    int radix2(double *realIO, double *imagIO) const;
    int radix2(double const *realIN, double *real, double *imag) const;
};

namespace { // anonymous namespace instead of static
    inline bool isPowOf2(int x) {
        return x > 0 && !(x & (x - 1)); // single bit set => power of 2
    }

    // floor(log(val)/log(2.0));
    inline unsigned int log2uint(unsigned int val) {
        unsigned int exp = -1;
        while (val != 0) {
            val >>= 1;
            ++exp;
        }
        return exp;
    }
}

template <typename T>
inline T nextpow2(T n)
{
    T next = 1;
    while (next < n && next>0)
        next <<= 1;
    return next;
}
