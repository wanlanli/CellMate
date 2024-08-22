import functools
import numpy as np

__all__ = ['check_nD']


def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,      # np.float128 ; doesn't exist on windows
    'G': np.complex128,   # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def _count_wrappers(func):
    """Count the number of wrappers around `func`."""
    unwrapped = func
    count = 0
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
        count += 1
    return count


def _get_stack_length(func):
    """Return function call stack length."""
    _func = func.__globals__.get(func.__name__, func)
    length = _count_wrappers(_func)
    return length


class _DecoratorBaseClass:
    """Used to manage decorators' warnings stacklevel.

    The `_stack_length` class variable is used to store the number of
    times a function is wrapped by a decorator.

    Let `stack_length` be the total number of times a decorated
    function is wrapped, and `stack_rank` be the rank of the decorator
    in the decorators stack. The stacklevel of a warning is then
    `stacklevel = 1 + stack_length - stack_rank`.
    """

    _stack_length = {}

    def get_stack_length(self, func):
        length = self._stack_length.get(func.__name__, _get_stack_length(func))
        return length


class deprecate_func(_DecoratorBaseClass):
    """Decorate a deprecated function and warn when it is called.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    deprecated_version : str
        The package version when the deprecation was introduced.
    removed_version : str
        The package version in which the deprecated function will be removed.
    hint : str, optional
        A hint on how to address this deprecation,
        e.g., "Use `skimage.submodule.alternative_func` instead."

    Examples
    --------
    >>> @deprecate_func(
    ...     deprecated_version="1.0.0",
    ...     removed_version="1.2.0",
    ...     hint="Use `bar` instead."
    ... )
    ... def foo():
    ...     pass

    Calling ``foo`` will warn with::

        FutureWarning: `foo` is deprecated since version 1.0.0
        and will be removed in version 1.2.0. Use `bar` instead.
    """

    def __init__(self, *, deprecated_version, removed_version=None, hint=None):
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version
        self.hint = hint

    def __call__(self, func):
        message = (
            f"`{func.__name__}` is deprecated since version "
            f"{self.deprecated_version}"
        )
        if self.removed_version:
            message += f" and will be removed in version {self.removed_version}."
        if self.hint:
            # Prepend space and make sure it closes with "."
            message += f" {self.hint.rstrip('.')}."

        stack_rank = _count_wrappers(func)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            # warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        # modify docstring to display deprecation warning
        doc = f'**Deprecated:** {message}'
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped
