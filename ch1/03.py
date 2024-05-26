def main():
    str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    str = str.replace(", "," ").replace(".","")
    str_list = str.split(" ")

    number_list=[]
    for i in range(len(str_list)):
        number_list.append(len(str_list[i]))

    print(number_list)



if __name__ == '__main__':
    main()