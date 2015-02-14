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
        return
    else
        nHalf = n >> 1

        # data reordering
        xEven = x[(r.start):2:(r.stop-1)]
        xOdd = x[(r.start+1):2:(r.stop)]
        rangeEven = (r.start):(r.start+nHalf-1)
        rangeOdd = (r.start+nHalf):(r.stop)
        x[rangeEven] = xEven
        x[rangeOdd] = xOdd

        # call sub-tasks
        subTwiddleBasis = twiddleBasis * twiddleBasis
        myfftTask(x, rangeEven, nHalf, subTwiddleBasis)
        myfftTask(x, rangeOdd, nHalf, subTwiddleBasis)

        # core calculation
        twiddle = twiddleBasis
        i = r.start
        j = i + nHalf
        xEven = x[i]
        xOdd = x[j]
        x[i] = xEven + xOdd
        x[j] = xEven - xOdd
        k = 1

        while k < nHalf
            i = i + 1
            j = j + 1
            xEven = x[i]
            xOdd = twiddle * x[j]
            x[i] = xEven + xOdd
            x[j] = xEven - xOdd
            twiddle = twiddle * twiddleBasis
            k = k + 1
        end
    end
end

end # module
