import re

def count_envs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    begins = re.findall(r'\\begin\{([^}]+)\}', content)
    ends = re.findall(r'\\end\{([^}]+)\}', content)
    
    print(f"File: {filename}")
    print(f"Begins: {len(begins)}")
    print(f"Ends: {len(ends)}")
    
    if len(begins) != len(ends):
        print("Mismatch found!")
        from collections import Counter
        b_counts = Counter(begins)
        e_counts = Counter(ends)
        for env in set(begins) | set(ends):
            if b_counts[env] != e_counts[env]:
                print(f"  Environment '{env}': begin={b_counts[env]}, end={e_counts[env]}")

if __name__ == "__main__":
    count_envs("paper.tex")
    count_envs("appendix.tex")
