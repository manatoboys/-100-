#jupyter notebookではdataが多過ぎて出力ができなかった。

with open("neko.txt.mecab", "r") as f:
    data = f.readlines()    #各行の読み込み
    data = [line.strip("\n") for line in data]  #全ての行から"\n"を排除

ans = []    #回答保存用配列

#各行をループ
for text in data:
    if text == "EOS":
        break
    dict = {}

    #空白の場合の処理
    if text[0] == " ":
        dict["surface"] = " "
        dict["base"] = " "
        temp = text.split("\t")
        print(temp)
        dict["pos"] =temp[0]
        dict["pos1"] = temp[1]
    #それ以外
    else:
        temp = text.split("\t")
        dict["surface"] = temp[0]
        dict["base"] = temp[1]
        dict["pos"] = temp[2]
        dict["pos1"] = temp[3]
    ans.append(dict)

print(ans)
