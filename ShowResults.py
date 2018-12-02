import sympy as sym

from testLogger import *
from ProjectConstants import *


def plot_tanh_function(test_log, test_number):
    def quantize(input_value, a_values, b_values, quantization_function):
        """Get the result of the hard quantization function derived from the
        soft one (the meaning of the name 'SoftToHardQuantization') and the
        codeword the input was sorted to

        Parameters
        ----------
            input_value : ndarray
                The quantizer input
            a_values : numpy.ndarray
                ai coefficients
            b_values : numpy.ndarray
                bi coefficients
            quantization_function : sympy.FunctionClass
                Lambdified quantization function

        Returns
        -------
            quantizedValue : int
                The output of the quantizer
            codeword : int
                The index of the corresponding codeword
        """
        output = np.empty([input_value.size, 2])
        for inputIndex in range(input_value.size):
            if input_value[inputIndex] <= b_values[0]:
                output[inputIndex, 0] = - sum(a_values)
                output[inputIndex, 1] = 0
            if input_value[inputIndex] > b_values[-1]:
                output[inputIndex, 0] = sum(a_values)
                output[inputIndex, 1] = len(b_values)
            for jj in range(len(b_values) - 1):
                if b_values[jj] < input_value[inputIndex] <= b_values[jj + 1]:
                    output[inputIndex, 0] = quantization_function((b_values[jj] + b_values[jj + 1]) / 2)
                    output[inputIndex, 1] = jj + 1
        return output

    a = test_log.aCoefs[test_number - 1][0]
    b = test_log.bCoefs[test_number - 1][0]
    c = test_log.cCoefs[test_number - 1][0][0]

    # Create symbolic variable x
    sym_x = sym.symbols('x')

    sym_tanh = a[0] * sym.tanh(c*(sym_x + b[0]))
    for ii in range(1, len(b)):
        sym_tanh = sym_tanh + a[ii] * sym.tanh(c*(sym_x + b[ii]))
    # Convert the symbolic functions to numpy friendly (for substitution)
    q = sym.lambdify(sym_x, sym_tanh, "numpy")

    space = 5

    x = np.linspace(b[0]-space, b[-1]+space, 10000)

    quantized = quantize(x, a, b, q)

    plt.plot(x, q(x))
    plt.plot(x, quantized[:, 0])
    plt.show()


# Set the test log
log = TestLogger(TEST_LOG_MAT_FILE)

# # Delete tests
# log.delete(1234567890)

# Show content of tests
log.content()

# Plot test log (the file provided for the theoretical bounds only)
log.plot(DATA_MAT_FILE[0])

# Plot soft and hard quantization functions of specific test
# plot_tanh_function(log, 31)
