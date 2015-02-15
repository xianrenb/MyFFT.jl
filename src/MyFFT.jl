module MyFFT

export myfft

function myfft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if (n == 0) || (n & (n-1) != 0)
        throw(ArgumentError)
    end

    r = 1:n
    twiddleBasis = e^(-2.0 * pi * im / n)
    myfftTask(x, r, n, twiddleBasis)
end

function myfftTask(x::AbstractArray{Complex{Float64}, 1}, r::Range, n::Integer, twiddleBasis::Complex{Float64})
    if n == 1
        return x[r.start]
    else
        nHalf = n >> 1

        # call sub-tasks
        s = step(r)
        sDouble = s << 1
        rangeEven = (r.start):(sDouble):(r.stop-s)
        rangeOdd = (r.start+s):(sDouble):(r.stop)
        subTwiddleBasis = twiddleBasis * twiddleBasis
        subEven = myfftTask(x, rangeEven, nHalf, subTwiddleBasis)
        subOdd = myfftTask(x, rangeOdd, nHalf, subTwiddleBasis)

        # core calculation
        result = zeros(Complex{Float64}, n)
        twiddle = twiddleBasis
        i = 1
        j = i + nHalf
        xEven = subEven[i]
        xOdd = subOdd[i]
        result[i] = xEven + xOdd
        result[j] = xEven - xOdd

        while i < nHalf
            i = i + 1
            j = j + 1
            xEven = subEven[i]
            xOdd = twiddle * subOdd[i]
            result[i] = xEven + xOdd
            result[j] = xEven - xOdd
            twiddle = twiddle * twiddleBasis
        end

        result
    end
end

end # module
