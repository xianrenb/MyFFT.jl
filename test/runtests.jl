using MyFFT
using Base.Test

v = rand(256) + im * rand(256)
@time v_myfft = myfft(v)
@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-11)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-11)
end

v = rand(256) + im * rand(256)
@time v_myfft = myfft(v)
@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-11)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-11)
end

v = rand(1024) + im * rand(1024)
@time v_myfft = myfft(v)
@time v_fft = fft(v)

for i = 1:1024
    @test_approx_eq_eps(real(v_myfft[i]), real(v_fft[i]), 1e-11)
    @test_approx_eq_eps(imag(v_myfft[i]), imag(v_fft[i]), 1e-11)
end
