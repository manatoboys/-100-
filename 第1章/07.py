def make_str(x, y, z):
    return f'{x}時の{y}は{z}'

def main():
    print(make_str(x = 12, y = "気温", z = 22.4))

if __name__ == "__main__":
    main()