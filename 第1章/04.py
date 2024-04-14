def main():
    str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
    formatted_str = str.replace(".","")
    str_list = formatted_str.split(" ")
    result = {}
    number_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]
    for i, str in enumerate(str_list):
        if (i+1) in number_list:
            result[str[:1]] = i
        else:
            result[str[:2]] = i
    print(result)
    return result

if __name__ == "__main__":
    main()

