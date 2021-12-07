from threading import Thread
from questionOne import main as mainOne
from questionTwo import main as mainTwo
from questionThree import main as mainThree


def main() -> bool:
    numThreads = 3
    results = [False] * numThreads
    threads = [None] * numThreads
    for i, func in enumerate([mainOne, mainTwo, mainThree]):
        threads[i] = (Thread(target=func, args=(results, i)))
        threads[i].start()

    for thread in threads:
        thread.join()

    return not (False in results)


if __name__ == '__main__':
    if not main():
        exit(1)
