// fftLite.cpp - The fftLite class.  See description in fftLite.hpp.
//
// by Jonathan Chappelow (chappjc)

#include <vector>
#include <cmath>
#include <cstdint>

#include "fftLite.hpp"

// Intel's SIMD complex number multiplication
#define USE_SSE3_COMPLEX_MULT 0

#if (USE_SSE3_COMPLEX_MULT == 1)
#include <pmmintrin.h>



typedef struct {
    double real;
    double img;
} complex_num;

// Multiplying complex numbers using SSE3 intrinsics
void multiply_SSE3(double xr, double xi, double yr, double yi,
    complex_num *z);
#endif

#define M_PI       3.14159265358979323846

fftLite::fftLite(size_t N_) : cosLUT(nullptr), sinLUT(nullptr), 
cosLUT_blu(nullptr), sinLUT_blu(nullptr),
revBits(nullptr), temps(nullptr), N(0), log2N(0)
{
    if (N_ == 0)
        return;

    // prepare for radix-2 Cooley-Tukey
    if (isPowOf2(N_))
    {
        M = 0;
        N = N_;
    }
    else { // prepare for Bluestein
        M = N_;
        N = nextpow2(2 * N_ + 1);

        cosLUT_blu = new double[M];
        sinLUT_blu = new double[M];

        for (int i = 0; i < M; ++i) {
            double temp = M_PI * (((uint64_t)i * i) % (2*M)) / M;
            //double temp = M_PI * i * i / M; // cos(temp), with large temp
            cosLUT_blu[i] = cos(temp);
            sinLUT_blu[i] = sin(temp);
        }

        temps = new double[4 * N];
    }

    // radix twiddle factors
    cosLUT = new double[N / 2];
    sinLUT = new double[N / 2];

    for (int i = 0; i < N / 2; ++i) {
        cosLUT[i] = cos(2 * M_PI * i / N);
        sinLUT[i] = sin(2 * M_PI * i / N);
    }

    // pre-compute bit reversal of position index
    log2N = log2uint(N);
    revBits = new int[N];
    for (int i = 0; i < N; ++i)
        revBits[i] = reverseBits(i, log2N);
}

fftLite::fftLite() : cosLUT(nullptr), sinLUT(nullptr), revBits(nullptr), temps(nullptr), 
N(0), log2N(0)
{
}

fftLite::~fftLite()
{
    delete[] cosLUT_blu;
    delete[] sinLUT_blu;
    delete[] cosLUT;
    delete[] sinLUT;
    delete[] revBits;
    delete[] temps;
}

unsigned int fftLite::reverseBits(unsigned int x, unsigned int n) // static
{
    unsigned int xrev = 0;
    for (unsigned int i = 0; i < n; ++i, x >>= 1)
        xrev = (xrev << 1) | (x & 1);
    return xrev;
}

int fftLite::fft(double *realIO, double *imagIO)
{
    if (N == 0)
        return -1;
    
    if (M == 0 /*(N & (N - 1)) == 0*/)  
        return radix2(realIO, imagIO);
    else // non-power-of-two length
        return bluestein(realIO, imagIO);
}

int fftLite::fft(double const *realIN, double *real, double *imag)
{
    if (N == 0)
        return -1;

    memcpy(real, realIN, N*sizeof(double));
    memset(imag, 0, N*sizeof(double));

    return fft(real, imag);
}

// in-place Bluesten algorithm.  See czt.m in MATLAB.
int fftLite::bluestein(double *realIO, double *imagIO) const
{
    int err;

    // to make VS2013 autovectorizer happy:
    const int _M = M, _N = N;
    const double *_cosLUT_blu = cosLUT_blu, *_sinLUT_blu = sinLUT_blu;

    // Premultiply data
    double *xr = temps, *xi = temps + N;
    for (int i = 0; i < _M; ++i) {
        xr[i] =  realIO[i] * _cosLUT_blu[i] + imagIO[i] * _sinLUT_blu[i];
        xi[i] = -realIO[i] * _sinLUT_blu[i] + imagIO[i] * _cosLUT_blu[i];
    }

    for (int i = _M; i < _N; ++i) {
        xr[i] = 0.0; xi[i] = 0.0;
    }
    
    double *yr = temps + 2*N, *yi = temps + 3*N;
    yr[0] = cosLUT_blu[0];
    yi[0] = sinLUT_blu[0];
    for (int i = 1; i < _M; ++i) {
        yr[i] = yr[N - i] = _cosLUT_blu[i];
        yi[i] = yi[N - i] = _sinLUT_blu[i];
    }

    for (int i = _M; i <= _N / 2; ++i) {
        yi[i] = yi[N - i] = yr[i] = yr[N - i] = 0.0;
    }

    // Fast convolution via FFT:
    // x = convolve(x,y)
    int err1, err2;
//#pragma omp parallel sections num_threads(4)
    {
        //#pragma omp section
        {err1 = radix2(xr, xi); }
        //#pragma omp section
        {err2 = radix2(yr, yi); }
    }
    err = err1 || err2;
    if (err) return err;
    //if (err = radix2(xr, xi)) return err;
    //if (err = radix2(yr, yi)) return err;

    // multiply 2 complex arrays: (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
#if (USE_SSE3_COMPLEX_MULT==1)
    for (int i = 0; i < N; ++i) {
        complex_num z;
        multiply_SSE3(xr[i], xi[i], yr[i], yi[i], &z);
        xr[i] = z.real;
        xi[i] = z.img;
    }
#else
    for (int i = 0; i < _N; ++i) {
        // 4 multiplications, 2 sums, 5 temporaries (1 explicit)
        double temp = xr[i] * yr[i] - xi[i] * yi[i]; // (ac - bd)
        xi[i] = xi[i] * yr[i] + xr[i] * yi[i];       // (ad + bc)
        xr[i] = temp;
        // 3 multiplications, 5 sums, 4 temporaries (2 explicit)
        //double ac = xr[i] * yr[i], bd = xi[i] * yi[i];
        //xi[i] = (xr[i] + xi[i]) * (yr[i] + yi[i]) - ac - bd;
        //xr[i] = ac - bd;
    }
#endif

    if (err = radix2(xi, xr)) return err; // ifft(xr,xi)

    for (int i = 0; i < _N; ++i) {
        xr[i] /= _N;
        xi[i] /= _N;
    }

    // Final Multiply
    for (int i = 0; i < _M; i++) {
        realIO[i] =  xr[i] * _cosLUT_blu[i] + xi[i] * _sinLUT_blu[i];
        imagIO[i] = -xr[i] * _sinLUT_blu[i] + xi[i] * _cosLUT_blu[i];
    }

    return 0;
}

int fftLite::radix2(double *realIO, double *imagIO) const
{
    if (!isPowOf2(N))
        return 1;
    
    // Bit-reversal permutation
    for (int i = 0; i < N; ++i) {
        int j = revBits[i];// reverseBits(i, log2N);
        if (j > i) {
            std::swap(realIO[i], realIO[j]);
            std::swap(imagIO[i], imagIO[j]);
        }
    }

    // Cooley-Tukey radix-2 decimation-in-time FFT
    for (int size = 2; size <= N; size *= 2) {
        int period = size / 2;
        int rate = N / size;
#pragma omp parallel for schedule(static) num_threads(2)
        for (int i = 0; i < N; i += size) {
            for (int j = i, k = 0; j < i + period; ++j, k += rate) {
                double treal =  realIO[j + period] * cosLUT[k] + imagIO[j + period] * sinLUT[k];
                double timag = -realIO[j + period] * sinLUT[k] + imagIO[j + period] * cosLUT[k];
                realIO[j + period] = realIO[j] - treal;
                imagIO[j + period] = imagIO[j] - timag;
                realIO[j] += treal;
                imagIO[j] += timag;
            }
        }
    }

    return 0;
}

// out of place, real-only input
int fftLite::radix2(double const *realIN, double *real, double *imag) const
{
    if (!isPowOf2(N))
        return 1;

    memcpy(real, realIN, N*sizeof(double));
    memset(imag, 0, N*sizeof(double));

    return radix2(real,imag);
}

// inverse is equivalent to swapping real and imaginary parts of input
int fftLite::ifft(double *realIO, double *imagIO) {
    return fft(imagIO, realIO);
}

// Multiplying complex numbers using SSE3 intrinsics
#if (USE_SSE3_COMPLEX_MULT == 1)
// from Intel's sample intrin_double_sample.c
void multiply_SSE3(double xr, double xi, double yr, double yi,
    complex_num *z)
{
    __m128d num1, num2, num3;

    // Duplicates lower vector element into upper vector element.
    //   num1: [x.real, x.real]

    num1 = _mm_loaddup_pd(&xr);

    // Move y elements into a vector
    //   num2: [y.img, y.real]

    num2 = _mm_set_pd(yi, yr);

    // Multiplies vector elements
    //   num3: [(x.real*y.img), (x.real*y.real)]

    num3 = _mm_mul_pd(num2, num1);

    //   num1: [x.img, x.img]

    num1 = _mm_loaddup_pd(&xi);

    // Swaps the vector elements
    //   num2: [y.real, y.img]

    num2 = _mm_shuffle_pd(num2, num2, 1);

    //   num2: [(x.img*y.real), (x.img*y.img)]

    num2 = _mm_mul_pd(num2, num1);

    // Adds upper vector element while subtracting lower vector element
    //   num3: [((x.real *y.img)+(x.img*y.real)),
    //          ((x.real*y.real)-(x.img*y.img))]

    num3 = _mm_addsub_pd(num3, num2);

    // Stores the elements of num3 into z

    _mm_storeu_pd((double *)z, num3);

}

// Multiplying complex numbers in C (reference)
void multiply_C(complex_num x, complex_num y, complex_num *z)
{
    z->real = (x.real*y.real) - (x.img*y.img);
    z->img = (x.img*y.real) + (y.img*x.real);
}

#endif

// DFT
void dft(const double *real, const double *imag,
    double *realOut, double *imagOut, int N) {

    // each frequency
    for (int k = 0; k < N; ++k) {
        double r = 0, i = 0;

        // each time/spacial point
        for (int t = 0; t < N; ++t) {
            double theta = 2 * M_PI * t * k / N;
            r +=  real[t] * cos(theta) + imag[t] * sin(theta);
            i += -real[t] * sin(theta) + imag[t] * cos(theta);
        }

        realOut[k] = r;
        imagOut[k] = i;
    }
}
