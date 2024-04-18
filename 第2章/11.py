FILE_PATH = "data/popular-names.txt"
Q11_PATH = "data/q11.txt"

def main():
    with open(FILE_PATH) as input_file:
        lines = input_file.readlines()

    with open(Q11_PATH, mode="w") as output_file:
        output_file.writelines([line.replace("\t"," ") for line in lines])

if __name__ == "__main__":
    main()
#P.10 明確な単語を選ぶ