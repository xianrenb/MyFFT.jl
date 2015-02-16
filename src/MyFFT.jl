module MyFFT

export myfft

function myfft(x::AbstractArray{Complex{Float64}, 1})
    n = size(x)[1]

    if (n == 0) || (n & (n-1) != 0)
        throw(ArgumentError)
    end

    r = 1:n
    twiddleBasis = e^(-2.0 * pi * im / n)
    twiddleTable = zeros(Complex{Float64}, n)
    twiddle = twiddleBasis

    for i in r
        if i > 2
            twiddle = twiddle * twiddleBasis
            twiddleTable[i] = twiddle
        elseif i == 1
            twiddleTable[i] = 1 + 0.0im
        else
            twiddleTable[i] = twiddle
        end
    end

    myfftTask(x, r, n, twiddleTable)
end

function myfftTask(x::AbstractArray{Complex{Float64}, 1}, r::Range, 
    n::Integer, twiddleTable::AbstractArray{Complex{Float64}, 1})
    if n == 1
        return x[r.start]
    else
        nHalf = n >> 1
        s = step(r)

        # call sub-tasks
        sDouble = s << 1
        rangeEven = (r.start):(sDouble):(r.stop-s)
        rangeOdd = (r.start+s):(sDouble):(r.stop)
        subEven = myfftTask(x, rangeEven, nHalf, twiddleTable)
        subOdd = myfftTask(x, rangeOdd, nHalf, twiddleTable)

        # core calculation
        result = zeros(Complex{Float64}, n)
        ttsize = size(twiddleTable)[1]
        twiddleIndex = 1
        i = 1
        j = i + nHalf

        while i <= nHalf
            xEven = subEven[i]
            xOdd = twiddleTable[twiddleIndex] * subOdd[i]
            result[i] = xEven + xOdd
            result[j] = xEven - xOdd
            twiddleIndex = twiddleIndex + s

            if twiddleIndex > ttsize
                twiddleIndex = twiddleIndex - ttsize
            end

            i = i + 1
            j = j + 1
        end

        result
    end
end

end # module
