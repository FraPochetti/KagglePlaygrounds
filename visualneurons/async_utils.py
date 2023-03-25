def run_generator_in_executor(loop, executor, generator):

    # We create an iterator
    iterator = iter(generator)

    # We need to define a special kind of next because generators
    # don't play well with async, and can't raise StopIteration
    def _async_next(iterator):

        try:
            return next(iterator)
        except StopIteration:
            raise AsyncStopIteration()

    while True:
        try:
            yield loop.run_in_executor(executor, _async_next, iterator)
        except AsyncStopIteration:
            break


class AsyncStopIteration(Exception):
    pass


def async_next(iterator):
    """We need to define a special kind of `next` because we can't
    raise StopIteration in async functions.

    """

    try:
        return next(iterator)
    except StopIteration:
        raise AsyncStopIteration()
