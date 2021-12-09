import pytest


def main():
    """
    Runs all the tests
    """
    import sys
    sys.path.append("/usr/src/Final")
    sys.path.append("/usr/src/Final/scripts")
    pytest.main(["--disable-pytest-warnings"])


if __name__ == '__main__':
    main()
