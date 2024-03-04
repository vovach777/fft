#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

namespace fft_implementation {
template <typename complex_t>
void fft0(int n, int s, bool eo, complex_t* x, complex_t* y)
// n  : sequence length
// s  : stride
// eo : x is output if eo == 0, y is output if eo == 1
// x  : input sequence(or output sequence if eo == 0)
// y  : work area(or output sequence if eo == 1)
{
    const int m = n / 2;
    const decltype(x->real()) theta0 = 2 * M_PI / n;

    if (n == 1) {
        if (eo)
            for (int q = 0; q < s; q++) y[q] = x[q];
    } else {
        for (int p = 0; p < m; p++) {
            const complex_t wp = complex_t(cos(p * theta0), -sin(p * theta0));
            for (int q = 0; q < s; q++) {
                const complex_t a = x[q + s * (p + 0)];
                const complex_t b = x[q + s * (p + m)];
                y[q + s * (2 * p + 0)] = a + b;
                y[q + s * (2 * p + 1)] = (a - b) * wp;
            }
        }
        fft0(n / 2, 2 * s, !eo, y, x);
    }
}

template <typename T>
auto fft(T&& data)  // Fourier transform
// n : sequence length
// x : input/output sequence
{
    auto n = data.size();
    auto x = data;
    auto y = data;
    fft0(n, 1, 0, x.data(), y.data());
    return x;
}

template <typename T>
auto ifft(T&& data)  // Fourier transform
// n : sequence length
// x : input/output sequence
{
    auto n = data.size();

    auto x = data;
    auto y = data;
    for (auto& v : x) v = std::conj(v);
    fft0(n, 1, 0, x.data(), y.data());
    auto t = x[0];
    t = 1.0 / n;
    for (auto& v : x) v = std::conj(v) * t;

    return x;
}

}  // namespace fft_implementation
using fft_implementation::fft;
using fft_implementation::ifft;

using number_t = float;
using complex_t = std::complex<number_t>;
using Array = std::vector<complex_t>;

#include <sstream>
template <typename T, typename D>
std::basic_ostream<T>& operator<<(std::basic_ostream<T>& os,
                                  const std::complex<D>& cpx) {
    std::stringstream ss;
    ss << "[" << std::fixed << std::setprecision(3) << cpx.real() << ", "
       << cpx.imag() << "]";
    os << ss.str();
    return os;
}

template <typename T, typename D>
std::basic_ostream<T>& operator<<(std::basic_ostream<T>& os,
                                  const std::vector<D>& vec) {
    for (auto v : vec) os << std::setw(15) << v;

    os << std::endl;

    return os;
}

int main() {
    auto data = Array(8);
    for (int i = 0; i < 8; ++i) {
        data[i] = i + 1;
    }
    std::cout << data;
    data = fft(data);
    std::cout << data;
    data = ifft(data);
    std::cout << data;
}
