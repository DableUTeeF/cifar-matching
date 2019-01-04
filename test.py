from data_loader import MatchingCifarLoader


def testLoader():
    a = MatchingCifarLoader('/home/palm/PycharmProjects/DATA/cifar10')
    test = a.testset
    t = test[1]
    return t


def testMultithread():
    import concurrent.futures
    import time

    def read_sensor_func(a1, a2, x):
        time.sleep(1)
        return (a1 + a2) * x

    def printtest(x):
        # time.sleep(0.1)
        print(x)

    def main():
        max_workers = 4
        my_list = [1, 12, 3, 99]
        arg1 = 8
        arg2 = 6
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            result = {executor.submit(read_sensor_func, arg1, arg2, x): x for x in my_list}
            executor.submit(printtest, 'test')
            for future in concurrent.futures.as_completed(result):
                if future.done():
                    print(future.result())

    main()


testMultithread()
