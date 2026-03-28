import functools


def square_decorator(title="Output", width=30):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print("=" * width)
            print(title.center(width))
            print("=" * width)

            print("\n")
            result = func(*args, **kwargs)
            print("\n")

            for _ in range(2):
                print("=" * width)

            return result

        return wrapper

    return decorator
