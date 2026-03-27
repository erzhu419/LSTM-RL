def find_non_ascii(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            for char_no, char in enumerate(line, 1):
                if ord(char) > 127:
                    print(f"Non-ASCII character '{char}' (U+{ord(char):04X}) at {filename}:{line_no}:{char_no}")

if __name__ == "__main__":
    find_non_ascii("paper.tex")
