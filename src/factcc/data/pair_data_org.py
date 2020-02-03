"""
Script for recreating the FactCC dataset from CNN/DM Story files.

CNN/DM Story files can be downloaded from https://cs.nyu.edu/~kcho/DMQA/

Unpaired FactCC data should be stored in a `unpaired_data` directory, 
CNN/DM data to be stored in a `cnndm` directory with `cnn` and `dm` sub-directories.
The script will save the recreated data in a `paired_data` directory.
"""
import json
import os


def parse_story_file(content):
    """
    Remove article highlights and unnecessary white characters.
    """
    content_raw = content.split("@highlight")[0]
    content = " ".join(filter(None, [x.strip() for x in content_raw.split("\n")]))
    return content


# Walk data sub-directories and recreate examples
for path, dirnames, filenames in os.walk("unpaired_data/"):
    print("Processed path:", path)

    for filename in filenames:
        full_path = os.path.join(path, filename)

        with open(full_path) as f:
            dataset = [json.loads(line) for line in f]

        for example in dataset:
            story_path = example["filepath"]
            with open(story_path) as f:
                story_content = f.read()
                example["text"] = parse_story_file(story_content)

        new_path = full_path.replace("unpaired_", "paired_")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        with open(new_path, "w") as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
