from lab0.data import eval_ap

if __name__ == '__main__':
    print(eval_ap([0, 0, 0, 1, 1, 1]))
    print(eval_ap([0, 0, 1, 0, 1, 1]))
    print(eval_ap([0, 1, 0, 1, 0, 1]))
    print(eval_ap([1, 0, 1, 0, 1, 0]))
