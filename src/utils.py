""" Helpful functions for debugging/logging and other stuff """

def print_encoding(model_inputs, indent=4):
    """ Taken from the colab """
    indent_str = " " * indent
    print("{")
    for k, v in model_inputs.items():
        print(indent_str + k + ":")
        print(indent_str + indent_str + str(v))
    print("}")