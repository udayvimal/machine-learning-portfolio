import os

exclude = {'chatbot_env', '__pycache__', '.git', '.venv'}

def list_structure(root='.', prefix=''):
    for item in sorted(os.listdir(root)):
        if item in exclude:
            continue
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print(f"{prefix}📁 {item}/")
            list_structure(path, prefix + '    ')
        else:
            print(f"{prefix}📄 {item}")

if __name__ == "__main__":
    list_structure()
