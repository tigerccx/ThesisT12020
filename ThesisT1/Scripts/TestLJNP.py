import numpy as np
if __name__ == '__main__':
    a = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]], dtype=int)
    print(a[..., 1].shape)

    a2 = np.array(a)

    w1 = np.where(a < 4)
    print(w1)
    print(w1[0])

    a3 = np.array(a2)
    print(a3[w1 and a3 <= 2].shape)
    a3[w1 and a3 <= 2] = 0
    print(a3)

    a4 = np.array(a2)
    print(a4[w1 or a4 <= 2].shape)
    a4[w1 or a4 <= 2] = 0
    print(a4)