import random

def shuffle(str: str):
    if len(str)>4:
        str_list = list(str)
        first = str_list.pop(0)
        last = str_list.pop(-1)
        random.shuffle(str_list)
        result = first + "".join(str_list) + last
        return result
    else:
        return str

def main(str: str):
    str_list = str.split()
    result_list = []
    for str in str_list:
        shuffled_str = shuffle(str)
        result_list.append(shuffled_str)
    result = " ".join(result_list)

    print(result)

if __name__ == "__main__":
    str = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    main(str = str)


