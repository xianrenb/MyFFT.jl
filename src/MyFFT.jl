module MyFFT

export myfft, myifft, myirealfft, myrealfft

function _myBluestein(x::AbstractArray{Complex{Float64}, 1}, 
    twiddleBasis::Complex{Float64})

    n = size(x)[1]
    m = 1

    while m < (n << 1) - 1
        m = m << 1
    end

    a = zeros(Complex{Float64}, m)
    b = zeros(Complex{Float64}, m)
    c = Array(Complex{Float64}, m)
    c[1] = 1 + 0.0im
    twiddleBasis2 = twiddleBasis * twiddleBasis
    twiddle = twiddleBasis
    twiddle2 = twiddle

    for i = 2:m
        c[i] = twiddle2
        # t^1, t^3, t^5, t^7, t^9,...
        twiddle = twiddle * twiddleBasis2
        # t^1, t^4, t^9, t^16, t^25,...
        twiddle2 = twiddle2 * twiddle
    end

    a[1:n] = x[1:n] .* c[1:n]
    b[1:n] = conj(c[1:n])
    i = 2
    j = m

    while i <= n
        b[j] = b[i]
        i = i + 1
        j = j - 1
    end

    copy((c .* _myconv(a, b))[1:n])
end

function _myconv(a::AbstractArray{Complex{Float64}, 1}, 
    b::AbstractArray{Complex{Float64}, 1})

    myifft(myfft(a) .* myfft(b))
end

function _myfftTask!(x::AbstractArray{Complex{Float64}, 1}, r::Range, 
    result::AbstractArray{Complex{Float64}, 1}, rResult::Range, n::Integer, 
    twiddleBasis::Complex{Float64})

    if n == 1
        result[rResult.start] = x[r.start]
    else
        nHalf = n >> 1

        # call sub-tasks
        s = step(r)
        sDouble = s << 1
        rEven = (r.start):(sDouble):(r.stop-s)
        rOdd = (r.start+s):(sDouble):(r.stop)
        rrEven = (rResult.start):(rResult.start+nHalf-1)
        rrOdd = (rResult.start+nHalf):(rResult.stop)
        subTwiddleBasis = twiddleBasis * twiddleBasis
        _myfftTask!(x, rEven, result, rrEven, nHalf, subTwiddleBasis)
        _myfftTask!(x, rOdd, result, rrOdd, nHalf, subTwiddleBasis)

        # core calculation
        twiddle = twiddleBasis
        i = rrEven.start
        j = rrOdd.start
        xEven = result[i]
        xOdd = result[j]
        result[i] = xEven + xOdd
        result[j] = xEven - xOdd

        while i < rrEven.stop
            i = i + 1
            j = j + 1
            xEven = result[i]
            xOdd = twiddle * result[j]
            result[i] = xEven + xOdd
            result[j] = xEven - xOdd
            twiddle = twiddle * twiddleBasis
        end
    end

    nothing
end

function myfft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if n == 0
        throw(ArgumentError)
    elseif n & (n-1) == 0
        r = 1:n
        twiddleBasis = e^(-2.0 * pi * im / n)
        result = Array(Complex{Float64}, n)
        _myfftTask!(x, r, result, r, n, twiddleBasis)
    else
        twiddleBasis = e^(-1.0 * pi * im / n)
        result = _myBluestein(x, twiddleBasis)
    end

    result
end

function myifft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if n == 0
        throw(ArgumentError)
    elseif n & (n-1) == 0
        r = 1:n
        twiddleBasis = e^(2.0 * pi * im / n)
        result = Array(Complex{Float64}, n)
        _myfftTask!(x, r, result, r, n, twiddleBasis)
    else
        twiddleBasis = e^(1.0 * pi * im / n)
        result = _myBluestein(x, twiddleBasis)
    end

    result/n
end

function myirealfft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if (n == 0) || (n % 2 != 0)
        throw(ArgumentError)
    end

    nHalf = n >> 1
    z = Array(Complex{Float64}, nHalf)
    twiddleBasis = e^(2.0 * pi * im / n)
    twiddle = twiddleBasis
    p = x[1]
    q = x[nHalf+1]
    xEven = 0.5 * (p + q)
    xOdd = 0.5 * (p - q)
    z[1] = xEven + 1im * xOdd

    for i = 1:nHalf-1
        p = x[i+1]
        q = x[nHalf+i+1]
        xEven = 0.5 * (p + q)
        xOdd = 0.5 * (p - q) * twiddle
        z[i+1] = xEven + 1im * xOdd
        twiddle = twiddle * twiddleBasis
    end

    z_ifft = myifft(z)
    result = Array(Float64, n)
    result[1:2:n-1] = real(z_ifft)
    result[2:2:n] = imag(z_ifft)
    result
end

function myrealfft(x::AbstractArray{Float64, 1})
    n = size(x)[1]

    if (n == 0) || (n % 2 != 0)
        throw(ArgumentError)
    end

    nHalf = n >> 1
    z = x[1:2:n-1] + 1im * x[2:2:n]
    z_fft = myfft(z)
    result = Array(Complex{Float64}, n)
    twiddleBasis = e^(-2.0 * pi * im / n)
    twiddle = -1im * twiddleBasis
    p = z_fft[1]
    q = conj(z_fft[1])
    result[1] = 0.5 * (p + q - 1im * (p - q))
    result[nHalf+1] = 0.5 * (p + q  + 1im * (p - q))
    i = 1
    j = nHalf - 1

    while i < nHalf
        p = z_fft[i+1]
        q = conj(z_fft[j+1])
        result[i+1] = 0.5 * (p + q  + twiddle * (p - q))
        result[n-i+1] = conj(result[i+1])
        twiddle = twiddle * twiddleBasis
        i = i + 1
        j = j - 1
    end

    result
end

end # module
