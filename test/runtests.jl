using MyFFT
using Base.Test

FFTW.set_num_threads(1)
fft([0.0im, 0.0im, 0.0im])
myfft([0.0im, 0.0im, 0.0im])
myifft([0.0im, 0.0im, 0.0im])
myrealfft([1.0, 1.0])
myirealfft([0.0im, 0.0im])

n = [1, 2, 4, 256, 512, 1024, 262144, 524288, 1048576, 3, 5, 7, 9, 253, 254, 255, 257, 258, 259]

for i in n
    println("N = $(i):")
    v = rand(i) + im * rand(i)
    println("myfft():")
    @time v_myfft = myfft(v)
    println("fft():")
    @time v_fft = fft(v)

    for j = 1:i
        @test_approx_eq_eps(real(v_myfft[j]), real(v_fft[j]), 1e-12 * i * log2(i))
        @test_approx_eq_eps(imag(v_myfft[j]), imag(v_fft[j]), 1e-12 * i * log2(i))
    end
end

n = [1, 2, 4, 256, 512, 1024, 253, 254, 255, 257, 258, 259]

for i in n
    println("N = $(i):")
    v = rand(i) + im * rand(i)
    println("myifft(myfft()):")
    @time v2 = myifft(myfft(v))

    for j = 1:i
        @test_approx_eq_eps(real(v), real(v2), 1e-12 * i * log2(i) ^ 2)
        @test_approx_eq_eps(imag(v), imag(v2), 1e-12 * i * log2(i) ^ 2)
    end
end

n = [2, 4, 6, 8, 256, 250, 252, 254, 258, 260, 262]

for i in n
    println("N = $(i):")
    v = rand(i)
    println("myrealfft():")
    @time v_realfft = myrealfft(v)
    println("fft():")
    @time v_fft = fft(v)

    for j = 1:i
        @test_approx_eq_eps(real(v_realfft), real(v_fft), 1e-12 * i * log2(i))
        @test_approx_eq_eps(imag(v_realfft), imag(v_fft), 1e-12 * i * log2(i))
    end
end

for i in n
    println("N = $(i):")
    v = rand(i)
    println("myirealfft(myrealfft()):")
    @time v2 = myirealfft(myrealfft(v))

    for j = 1:i
        @test_approx_eq_eps(real(v), real(v2), 1e-12 * i * log2(i) ^ 2)
        @test_approx_eq_eps(imag(v), imag(v2), 1e-12 * i * log2(i) ^ 2)
    end
end
