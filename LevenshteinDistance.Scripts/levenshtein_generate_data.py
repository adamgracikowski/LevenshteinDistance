import argparse
import random
import string

def generate_random_word(length):
    """Generates a random string of the specified length."""
    return ''.join(random.choices(string.ascii_letters, k=length))

def generate_text_file(file_name, n, m):
    """Generates a text file."""
    sourceWord = generate_random_word(n)
    targetWord = generate_random_word(m)

    with open(file_name, 'w') as file:
        file.write(f"{n} {m}\n")
        file.write(f"{sourceWord}\n")
        file.write(f"{targetWord}\n")

def generate_binary_file(file_name, n, m):
    """Generates a binary file."""
    sourceWord = generate_random_word(n).encode('utf-8')
    targetWord = generate_random_word(m).encode('utf-8')

    with open(file_name, 'wb') as file:
        file.write(n.to_bytes(4, byteorder='little'))
        file.write(m.to_bytes(4, byteorder='little'))
        file.write(sourceWord)
        file.write(targetWord)

def main():
    parser = argparse.ArgumentParser(description="Generates an input file in text or binary format.")
    parser.add_argument("data_format", choices=["bin", "txt"], help="Data format: bin for binary, txt for text.")
    parser.add_argument("n", type=int, help="Length of the source word.")
    parser.add_argument("m", type=int, help="Length of the target word.")
    parser.add_argument("output_file", help="Name of the output file.")

    args = parser.parse_args()

    if args.data_format == 'txt':
        generate_text_file(args.output_file, args.n, args.m)
        print(f"Text file '{args.output_file}' generated successfully with words of lengths {args.n} and {args.m}.")
    elif args.data_format == 'bin':
        generate_binary_file(args.output_file, args.n, args.m)
        print(f"Binary file '{args.output_file}' generated successfully with words of lengths {args.n} and {args.m}.")

if __name__ == "__main__":
    main()