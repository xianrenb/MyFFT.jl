module MyFFT

export myfft, myifft

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

        result
    end
end

function myfft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if (n == 0) || (n & (n-1) != 0)
        throw(ArgumentError)
    end

    r = 1:n
    twiddleBasis = e^(-2.0 * pi * im / n)
    result = Array(Complex{Float64}, n)
    _myfftTask!(x, r, result, r, n, twiddleBasis)
    result
end

function myifft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if (n == 0) || (n & (n-1) != 0)
        throw(ArgumentError)
    end

    r = 1:n
    twiddleBasis = e^(2.0 * pi * im / n)
    result = Array(Complex{Float64}, n)
    _myfftTask!(x, r, result, r, n, twiddleBasis)
    result/n
end

end # module
