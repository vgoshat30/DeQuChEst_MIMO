def codebook_uniform(input_variance, m):
    """Simple variance based uniform quantizer


    Parameters
    ----------
        input_variance : float
            The variance of the input x in R^1 to the scalar quantizers.
        m : int
            The number of codewords expected in the output.

    Returns
    -------
        codewords : tuple
            codewords of shape `(M,1)`: tensor containing the output codewords
            of the quantizer. This will be the codebook dictionary.
    """
    # We first divide the R plane into two regions, where the codewords will
    # lay. we use the symmetry of this quantizer
    codebook = []
    lower_bound = -3*input_variance
    upper_bound = 3*input_variance
    dx = (upper_bound - lower_bound)/(m + 1)
    for ii in range(1, m + 1):
        codebook.append(lower_bound + ii*dx)
    return tuple(codebook)


def get_optimal_word(x, codebook):
    """Return the matching codeword to the input

    Parameters
    ----------
        x : float
            The input x in R^1 to the scalar quantizers.
        codebook : tuple
            The set of words available as the output.

    Returns
    -------
        quantized_word : float
            The output word of the quantizer.
        codeword_idx : int
            The
    """
    codeword_idx = 0
    quantized_word = codebook[0]
    # If the input is bigger than the highest codeword, return the codeword
    if x > codebook[-1]:
        quantized_word = codebook[-1]
        codeword_idx = len(codebook) - 1
        return quantized_word, codeword_idx
    for ii in range(0, len(codebook) - 1):
        # If the input is between the lowest and highest codeword -
        # get the optimal word
        if codebook[ii] < x < codebook[ii + 1]:
            if x <= ((codebook[ii + 1] + codebook[ii]) / 2):
                quantized_word = codebook[ii]
                codeword_idx = ii
            else:
                # If the input is smaller than the lowest codeword,
                # return the codeword
                quantized_word = codebook[ii + 1]
                codeword_idx = ii + 1
            return quantized_word, codeword_idx
    return quantized_word, codeword_idx
