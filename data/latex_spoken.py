from datasets import load_dataset, DatasetDict

data_dict = load_dataset("json", data_files="data.json")

trainval_test = data_dict["train"].train_test_split(test_size=0.1, seed=2112005)
train_val = trainval_test["train"].train_test_split(test_size=10000, seed=2112005)

final_dataset = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": trainval_test["test"]
})
print(final_dataset)

final_dataset.push_to_hub("Last-Bullet/LaTeX_Spoken")