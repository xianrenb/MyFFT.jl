using MyFFT
using Base.Test

v = rand(256) + im * rand(256)

@time begin
    v_myfft = copy(v)
    myfft(v_myfft)
end

@time v_fft = fft(v)

for i = 1:256
    @test_approx_eq(real(v_myfft[i]), real(v_fft[i]))
    @test_approx_eq(imag(v_myfft[i]), imag(v_fft[i]))
end
