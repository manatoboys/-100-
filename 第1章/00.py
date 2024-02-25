def main():
    str = "stressed"
    
    str_list = list(str)
    str_list.reverse()
    str_rev = "".join(str_list)
    
    print(str_rev)
    return str_rev

if __name__ == '__main__':
    main()