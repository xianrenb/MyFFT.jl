using MyFFT
using Base.Test

v = rand(256) + im * rand(256)

@time begin
    v_myfft = copy(v)
    myfft(v_myfft)
end

@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-12)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-12)
end

v = rand(256) + im * rand(256)

@time begin
    v_myfft = copy(v)
    myfft(v_myfft)
end

@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-12)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-12)
end

v = rand(1024) + im * rand(1024)

@time begin
    v_myfft = copy(v)
    myfft(v_myfft)
end

@time v_fft = fft(v)

for i = 1:1024
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-12)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-12)
end
