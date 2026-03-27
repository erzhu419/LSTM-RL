def check_bib_braces(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stack = []
    for i, char in enumerate(content):
        if char == '{':
            stack.append(i)
        elif char == '}':
            if not stack:
                print(f"Unmatched '}}' at index {i}")
            else:
                stack.pop()
    
    if stack:
        for pos in stack:
            print(f"Unmatched '{{' at index {pos}")
            # Show context
            start = max(0, pos - 20)
            end = min(len(content), pos + 50)
            print(f"Context: {content[start:end]}")

if __name__ == "__main__":
    check_bib_braces("ensemble_refs.bib")
