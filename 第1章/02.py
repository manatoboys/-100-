def main():
    str1 = "パトカー"
    str2 = "タクシー"
    str = ""

    for i in range(4):
        str += str1[i]
        str += str2[i]

    print(str)
    return str

if __name__ == '__main__':
    main()