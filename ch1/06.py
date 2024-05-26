def bi_gram(str: str):
    list_moji = list(str)
    moji_bigram = []
    for i in range(len(list_moji)- 1):
        moji = ""
        for j in range(i,i+2):
            moji += list_moji[j]
        moji_bigram.append(moji)
    return moji_bigram

def main():
    str1 = "paraparaparadise"
    bi_gram1 = bi_gram(str1)

    str2 = "paragraph"
    bi_gram2 = bi_gram(str2)

    set_str1 = set(bi_gram1)
    set_str2 = set(bi_gram2)

    union = set_str1 | set_str2
    intersection = set_str1 & set_str2
    difference = set_str1 - set_str2

    print(f'和集合: {union}')
    print(f'積集合: {intersection}')
    print(f'差集合: {difference}')
    
    In_str1 = "se" in set_str1
    In_str2 = "se" in set_str2

    print(f'se in "{str1}"?: {In_str1}')
    print(f'se in "{str2}"?: {In_str2}')

if __name__ == "__main__":
    main()