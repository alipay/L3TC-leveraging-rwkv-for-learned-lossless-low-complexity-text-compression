from arithmeticcoding import *


def main():
    string_to_be_coding = "ABCDEdkkjkljkljkkjkkkklmkmmmmmmmmmmmmaaaasldkkjlfjklajflksjlkfjskljklfjskljfdlskjfklsjlfjlsaacc"
    string_dict = dict()
    for symbol in string_to_be_coding:
        if symbol in string_dict:
            string_dict[symbol]["count"] += 1
        else:
            string_dict[symbol] = {"count": 1}

    char_table = []
    freq_table = []
    for key_id, key in enumerate(string_dict):
        char_table.append(key)
        freq_table.append(string_dict[key]["count"])
        string_dict[key]["idx"] = key_id

    input_file = "./compress_code.bin"

    # build the input stream
    bitin = BitInputStream(open(input_file, mode='rb'))

    # build arithmetic decoder
    dec = ArithmeticDecoder(bitin)

    output_string = ""
    for i in range(len(string_to_be_coding)):
        # build frequency table for coding
        # freq_table[-1] = freq_table[-1] + 1
        freq = SimpleFrequencyTable(freq_table)

        symbol = dec.read(freq)
        output_string += char_table[symbol]

    print(output_string)
    print(string_to_be_coding)
    print(output_string == string_to_be_coding)



if __name__ == "__main__":
    main()
