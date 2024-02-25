import sys
FILE_PATH = "data/popular-names.txt"

def write_lines(n: int):
    with open(FILE_PATH) as file:
        lines = file.readlines()
    
    for i in range(len(lines)-n,len(lines)):
        print(lines[i], end ="")

def main():
    args = sys.argv
    n = int(args[1])
    write_lines(n=n)

if __name__ == "__main__":
    main()