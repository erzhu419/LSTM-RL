def check_braces(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stack = []
    line_no = 1
    char_no = 0
    for i, char in enumerate(content):
        if char == '\n':
            line_no += 1
            char_no = 0
        else:
            char_no += 1
            
        if char == '{':
            stack.append((line_no, char_no))
        elif char == '}':
            if not stack:
                print(f"Unmatched '}}' at line {line_no}, col {char_no}")
            else:
                stack.pop()
                
    for line, col in stack:
        print(f"Unmatched '{{' at line {line}, col {col}")

if __name__ == "__main__":
    print("Checking paper.tex...")
    check_braces("paper.tex")
    print("Checking appendix.tex...")
    check_braces("appendix.tex")
