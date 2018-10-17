def convert_dict_key(x):
    d = {}
    y = []
    cur = 0

    for i in x:
        if i not in d.keys():
            d[i] = cur
            cur += 1

        y.append(d[i])

    return y
