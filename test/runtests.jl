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
    myfftClearCache()
    gc()
    @time v_myfft = myfft(v)
    gc()
    @time v_myfft = myfft(v)
    println("fft():")
    gc()
    @time v_fft = fft(v)
    gc()
    @time v_fft = fft(v)

    for j = 1:i
        eps_fft = 2^0.5 * eps() * 2.0 * ceil(log2(2*i-1))^2.0
        eps_myfft = 2^0.5 * eps() * 2.0 * nextpow2(2*i-1)^2.0 * ceil(log2(2*i-1))^2.0
        eps_total = eps_fft + eps_myfft
        @test_approx_eq_eps(real(v_myfft[j]), real(v_fft[j]), eps_total)
        @test_approx_eq_eps(imag(v_myfft[j]), imag(v_fft[j]), eps_total)
    end
end

n = [1, 2, 4, 256, 512, 1024, 253, 254, 255, 257, 258, 259]

for i in n
    println("N = $(i):")
    v = rand(i) + im * rand(i)
    println("myifft(myfft()):")
    myfftClearCache()
    gc()
    @time v2 = myifft(myfft(v))

    for j = 1:i
        eps_fft = 2^0.5 * eps() * 4.0 * ceil(log2(2*i-1))^4.0
        eps_myfft = 2^0.5 * eps() * 4.0 * nextpow2(2*i-1)^4.0 * ceil(log2(2*i-1))^4.0
        eps_total = eps_fft + eps_myfft
        @test_approx_eq_eps(real(v[j]), real(v2[j]), eps_total)
        @test_approx_eq_eps(imag(v[j]), imag(v2[j]), eps_total)
    end
end

n = [2, 4, 6, 8, 256, 250, 252, 254, 258, 260, 262, 1, 3, 5, 7, 251, 253, 255, 257, 259, 261]

for i in n
    println("N = $(i):")
    v = rand(i)
    println("myrealfft():")
    myfftClearCache()
    gc()
    @time v_realfft = myrealfft(v)
    gc()
    @time v_realfft = myrealfft(v)
    println("fft():")
    gc()
    @time v_fft = fft(v)
    gc()
    @time v_fft = fft(v)

    for j = 1:i
        eps_fft = 1.0 * eps() * 2.0 * ceil(log2(2*i-1))^2.0
        eps_myfft = 1.0 * eps() * 2.0 * nextpow2(2*i-1)^2.0 * ceil(log2(2*i-1))^2.0
        eps_total = eps_fft + eps_myfft
        @test_approx_eq_eps(real(v_realfft[j]), real(v_fft[j]), eps_total)
        @test_approx_eq_eps(imag(v_realfft[j]), imag(v_fft[j]), eps_total)
    end
end

for i in n
    println("N = $(i):")
    v = rand(i)
    println("myirealfft(myrealfft()):")
    myfftClearCache()
    gc()
    @time v2 = myirealfft(myrealfft(v))

    for j = 1:i
        eps_fft = 1.0 * eps() * 4.0 * ceil(log2(2*i-1))^4.0
        eps_myfft = 1.0 * eps() * 4.0 * nextpow2(2*i-1)^4.0 * ceil(log2(2*i-1))^4.0
        eps_total = eps_fft + eps_myfft
        @test_approx_eq_eps(v[j], v2[j], eps_total)
    end
end

println("BigFloat:")
x = im * convert(BigFloat, 0)
gc()
@time myfft([x, x, x])
gc()
@time myfft([x, x, x])
