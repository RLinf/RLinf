import json

DATA_PATH = "/mnt/public/yushi/Multiagent/sft_datasets/100/sft_train.jsonl"


# [0, 3, 3, 8, 7, 10, 3, 10, 3, 4, 9, 3, 6, 9, 6, 7, 7, 3, 2, 10, 5, 10, 10, 4, 10, 9, 9, 7, 10, 10, 3, 10, 8, 9, 9, 2, 3, 10, 3, 10, 3, 3, 10, 9, 7, 7, 8, 5, 3, 7, 9, 9, 5, 3, 4, 8, 4, 3, 10, 2, 8, 10, 6, 10, 6, 4, 9, 5, 6, 10, 4, 10, 16, 4, 10, 10, 10, 7, 10, 7, 8, 6, 10, 6, 9, 10, 4, 9, 10, 4, 5, 6, 8, 7, 9, 7] 16

def read_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


if __name__ == "__main__":
    data = read_jsonl(DATA_PATH)
    print(f"Total samples: {len(data)}")
    print(f"Keys: {list(data[0].keys())}")
    print()

    # Print a few examples
    for i, sample in enumerate(data[:3]):
        print(f"--- Sample {i} ---")
        print(f"  instance_id: {sample['instance_id']}")
        print(f"  reward:      {sample['reward']}")
        print(f"  input:       {sample['input'][:100]}...")
        print(f"  output:      {sample['output'][:100]}...")
        print()

# len = []
# num = 0
# last = -1
# for id, d in enumerate(data):
#     if 'You are a main-agent working' in d['input']:
#         print(id, d['reward'])
#         if d['reward'] != last:
#             len.append(num)
#             num = 1
#             last = d['reward']
#         else:
#             num +=1
#             if num == 16:
#                 print("here", id, d['reward'])
# print(len, max(len))
