

# Prints lines from a file.
def printLines(file, n=10):
    """Print n number of lines from a file

    Args:
        file (string): Path to the file
        n (int, optional): Number of lines to print. Defaults to 10.
    """
    with open(file, 'rb') as f:
        lines = f.readlines()
    for line in lines[:n]:
        print(line)
