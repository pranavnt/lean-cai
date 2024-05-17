import lean_dojo
import json

BENCHMARK_DIR = "./leandojo_benchmark_4/random"

train_data = json.load(open(f"{BENCHMARK_DIR}/train.json", "r"))
test_data = json.load(open(f"{BENCHMARK_DIR}/test.json", "r"))
val_data = json.load(open(f"{BENCHMARK_DIR}/val.json", "r"))

print(val_data[0])
