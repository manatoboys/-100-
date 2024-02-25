def main(n: int):
    str = "I am an NLPer"
    list_word = str.split(" ")
    list_moji = list(str.replace(" ",""))
    word_ngram = []
    moji_ngram = []
    for i in range(len(list_word)- n + 1):
        word_ngram.append([list_word[i],list_word[i+1]])

    for i in range(len(list_moji)- n + 1):
        moji = ""
        for j in range(i,i+n):
            moji += list_moji[j]
        moji_ngram.append(moji)

    print(f'単語{n}-gram: {word_ngram}')
    print(f'文字{n}-gram: {moji_ngram}')

if __name__ == "__main__":
    main(n = 2)
