using MyFFT
using Base.Test

println("N = 256:")
v = rand(256) + im * rand(256)
println("myfft():")
@time v_myfft = myfft(v)
println("fft():")
@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-12 * 256 * log2(256))
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-12 * 256 * log2(256))
end

n = [1, 2, 4, 256, 1024, 262144, 1048576, 3, 5, 7, 9, 253, 254, 255, 257, 258, 259]

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

n = [1, 2, 256, 1024, 253, 254, 255, 257, 258, 259]

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
