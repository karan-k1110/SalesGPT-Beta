import json

SQS_SIZE = 256000

config_path = "/Users/karan/Developer/Personal/projects/SalesGPT-Beta/examples/test.json"


def check():
    try:
        with open(config_path, "r", encoding="UTF-8") as f:
            test_file = json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the config file {config_path}.")

    payload = test_file["records"]
    print(f''' payload size = {len(json.dumps(payload))}''')
    chunks = [payload[x:x + 200] for x in range(0, len(payload), 200)]
    for chunk in chunks:

        print(len(chunks))

if __name__ == "__main__":
    check()
