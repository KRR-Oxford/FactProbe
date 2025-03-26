import gzip 
import json 
from tqdm import tqdm 
import sys 

fn = sys.argv[1]
output_name = sys.argv[2]

print(f"Reading from {fn} and saving to {output_name}")

total_str = ""


with gzip.open(fn, mode="rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        line_text = row["text"]
        line_text = line_text.replace('\0', '').replace('\x00', '')
        total_str += line_text

if len(total_str) > 9e8: # approx 1 GB text file
    sum_len = 0
    n = len(total_str)
    partitions = n // int(9e8) + 2
    print(f"Number of partitions: {partitions}")
    partition_size = n // partitions
    print(f"Partition length: {partition_size}")

    for i in range(partitions):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == partitions - 1:
            end = n
        partition_text = total_str[start:end]
        text_len = end - start
        sum_len += text_len
        print(f"Partition {i} length: {text_len}")

        partition_file = output_name.split(".txt")[0] + f"_{i}.txt"
        with open(partition_file, mode="wt", encoding="utf-8") as f:
            f.write(partition_text)
        print(f"Partition {i} saved to {partition_file}")

    print(f"Total length of text: {n}")
    print(f"Total length of partitions: {sum_len}")
    assert n == sum_len

else:
    with open(output_name, mode="wt", encoding="utf-8") as f:
        f.write(total_str)

