FILE_PATH = "data/popular-names.txt"

def main():
    with open(FILE_PATH) as f:
        cnt = 0
        for row in f:
            cnt += 1
    print(cnt)

if __name__ == "__main__":
    main()

