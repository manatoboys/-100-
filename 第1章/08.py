def cipher(str: str):
    moji = ""
    for i in range(len(str)):
        if str[i].islower():
            moji += chr(219 - ord(str[i]))
        else:
            moji += str[i]

    print(moji)

def main():
    cipher("This is Japanese guy!")

if __name__ == "__main__":
    main()
