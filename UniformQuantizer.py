def codebook_uniform(input_variance, M):
    """Simple variance based uniform quantizer


    Parameters
    ----------
        input_variance : float
            The variance of the input x in R^1 to the scalar quantizers.
        M : int
            The number of codewords expected in the output.

    Returns
    -------
        codewords : tuple
            codewords of shape `(M,1)`: tensor containing the output codewords
            of the quantizer. This will be the codebook dictionary.
    """
    # We first divide the R plane into two regions, where the codewords will
    # lay, we use the symmetricity of this quantizers
    codebook = []
    LowerBound = -3*input_variance
    UpperBound = 3*input_variance
    dx = (UpperBound - LowerBound)/(M+1)
    for ii in range(1, M+1):
        codebook.append(LowerBound + ii*dx)
    return tuple(codebook)


def get_optimal_word(input, codebook):
    """Return the matching codeword to the input

    Parameters
    ----------
        input : float
            The input x in R^1 to the scalar quantizers.
        codebook : tuple
            The set of words avaliable as the output.

    Returns
    -------
        qunatized_word : float
            The output word of the quantizer.
        codewordIdx : int
            The
    """
    codewordIdx = 0
    qunatized_word = codebook[0]
    # If the input is bigger than the highest codeword, return the codeword
    if input > codebook[-1]:
        qunatized_word = codebook[-1]
        codewordIdx = len(codebook) - 1
        return qunatized_word, codewordIdx
    for ii in range(0, len(codebook) - 1):
        # If the input is between the lowest and highest codeword -
        # get the optimal word
        if(input > codebook[ii] and input < codebook[ii + 1]):
            if input <= ((codebook[ii + 1] + codebook[ii])/2):
                qunatized_word = codebook[ii]
                codewordIdx = ii
            else:
                # If the input is smaller than the lowest codeword,
                # return the codeword
                qunatized_word = codebook[ii + 1]
                codewordIdx = ii + 1
            return qunatized_word, codewordIdx
    return qunatized_word, codewordIdx
