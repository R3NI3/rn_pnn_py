import gzip

def read_dataset(path = str):
    with gzip.open(path, 'rb') as f:
        file_content = f.readlines()

    return file_content

if __name__ == '__main__':
    content = read_dataset('Databases/kddcup.data.gz')

    print content[0]
    print content[1]