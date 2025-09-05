def read_lines(filename):
    """Reads the expression in the given file and breaks it into a list of 
    lines, each line representing a term in the expression to be summed.
    """
    num_unpaired_brackets = 0
    next_line = []
    lines = []
    
    with open(filename, 'r') as file:
        while True:    
            char = file.read(1)
            if not char:
                lines.append(''.join(next_line))
                break
            
            next_line.append(char)
            
            if char == '(':
                num_unpaired_brackets += 1
            elif char == ')':
                num_unpaired_brackets -= 1
                
            if char == ' ' and num_unpaired_brackets == 0:
                lines.append(''.join(next_line))
                next_line = []
    
    return lines


def get_terms(lines):
    """Breaks the given list of lines (ie terms to be summed in an expression)
    into two lists: a list containing the terms to be summed and a list of 
    leading signs for each term (excluding the first term).
    """
    terms = []
    signs = []
    
    for line in lines:
        if line in ["+ ", "- "]:
            signs.append(1 if line == "+ " else -1)
        
        terms.append(line)
        
    return signs, terms
