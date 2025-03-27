import requests
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_github_tree(repo_owner, repo_name, branch='main'):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

def generate_markdown_tree(tree_data, max_depth=3):
    paths = [item['path'] for item in tree_data['tree']]
    paths.sort()
    
    output = ["📦 Repository Structure"]
    for path in paths:
        depth = path.count('/')
        if depth > max_depth:
            continue
            
        indent = '    ' * depth
        base = path.split('/')[-1]
        
        if depth == 0:
            line = f"├── 📂 {base}"
        else:
            line = f"{indent}├── {'📂' if '/' in path else '📜'} {base}"
        
        output.append(line)
    
    return "```markdown\n" + "\n".join(output) + "\n```"

# Configuration
REPO_OWNER = "Ellaermeg"
REPO_NAME = "Eliah-Masters"
BRANCH = "Working"

try:
    tree_data = get_github_tree(REPO_OWNER, REPO_NAME, BRANCH)
    markdown_tree = generate_markdown_tree(tree_data, max_depth=3)
    
    with open("REPOSITORY_STRUCTURE.md", "w", encoding='utf-8') as f:
        f.write(markdown_tree)
    print("Successfully generated repository structure!")
    print("Saved to REPOSITORY_STRUCTURE.md")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("Possible solutions:")
    print("1. Check your internet connection")
    print("2. Verify the repository is public")
    print("3. Try running in VS Code or another modern terminal")