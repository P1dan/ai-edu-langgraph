import os

def print_tree(root_dir, prefix=""):
    entries = sorted(os.listdir(root_dir))
    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree("src")
