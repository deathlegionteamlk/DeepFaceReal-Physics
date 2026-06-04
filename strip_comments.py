import ast, re, os

def strip_line(line):
    in_dq, in_sq, esc = False, False, False
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if esc:
            out.append(c)
            esc = False
            i += 1
            continue
        if c == '\\' and (in_dq or in_sq):
            out.append(c)
            esc = True
            i += 1
            continue
        if c == '"' and not in_sq:
            in_dq = not in_dq
            out.append(c)
            i += 1
            continue
        if c == "'" and not in_dq:
            in_sq = not in_sq
            out.append(c)
            i += 1
            continue
        if c == '#' and not in_dq and not in_sq:
            break
        out.append(c)
        i += 1
    return ''.join(out)

def remove_docstrings(code):
    """Remove standalone docstring expressions but keep string literals in assignments."""
    # We'll use AST to identify docstrings
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Constant, ast.Str))):
                if isinstance(node.body[0].value, ast.Constant):
                    if isinstance(node.body[0].value.value, str):
                        docstring_nodes.add(node.body[0].lineno)
    
    if not docstring_nodes:
        return code
    
    lines = code.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line_num = i + 1
        stripped = lines[i].strip()
        
        if line_num in docstring_nodes and (stripped.startswith('"""') or stripped.startswith("'''")):
            in_dq = stripped.startswith('"""')
            if (in_dq and stripped.endswith('"""') and len(stripped) > 3) or \
               (not in_dq and stripped.endswith("'''") and len(stripped) > 3):
                i += 1
                continue
            
            i += 1
            while i < len(lines):
                if in_dq and '"""' in lines[i]:
                    i += 1
                    break
                if not in_dq and "'''" in lines[i]:
                    i += 1
                    break
                i += 1
            continue
        
        new_lines.append(lines[i])
        i += 1
    
    return '\n'.join(new_lines)

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        return False, f"Read error: {e}"
    
    original = content
    
    # Step 1: Remove docstrings
    content = remove_docstrings(content)
    
    # Step 2: Remove inline comments
    lines = content.split('\n')
    lines = [strip_line(l) for l in lines]
    
    # Step 3: Clean up blank lines
    result = []
    prev_blank = False
    for l in lines:
        is_blank = len(l.strip()) == 0
        if is_blank and prev_blank:
            continue
        result.append(l.rstrip())
        prev_blank = is_blank
    
    content = '\n'.join(result)
    
    # Step 4: Verify syntax
    try:
        ast.parse(content)
    except SyntaxError as e:
        return False, f"Syntax error after stripping: {e}"
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    return True, None

base = '/app/real_time_deepfake_0851'
files = [
    'app.py', 'api.py',
    'core/__init__.py', 'core/face_swapper.py',
    'core/webcam_pipeline.py', 'core/character_manager.py',
    'core/llm_character.py', 'core/lip_sync.py'
]

all_ok = True
for fname in files:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f"MISSING: {fname}")
        continue
    
    with open(path, 'r') as fh:
        before = fh.read()
    
    ok, err = process_file(path)
    if ok:
        print(f"OK: {fname}")
    else:
        print(f"FAIL: {fname} - {err}")
        all_ok = False
        with open(path, 'w') as fh:
            fh.write(before)

if all_ok:
    print("\nAll files cleaned successfully!")
else:
    print("\nSome files failed!")
    exit(1)