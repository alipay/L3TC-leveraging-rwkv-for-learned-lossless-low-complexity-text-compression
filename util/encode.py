from arithmeticcoding import *
import base64


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
    
    output_file = "./compress_code.bin"

    # build the output stream
    bitout = BitOutputStream(open(output_file, "wb+"))

    # build arithmetic encoder
    enc = ArithmeticEncoder(bitout)

    print("".join(char_table))
    print(freq_table)
    freq = SimpleFrequencyTable(freq_table)

    # encoding string
    for symbol in string_to_be_coding:
        symbol_id = string_dict[symbol]["idx"]
        
        # build frequency table for coding
        # freq_table[-1] = freq_table[-1] + 1
        
        # import ipdb; ipdb.set_trace()

        enc.write(freq, symbol_id)

    enc.finish()
    bitout.close()
    
    
if __name__ == "__main__":
    main()