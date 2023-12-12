from datasets import load_dataset, Dataset
from multiprocessing import Pool
import string
import math
from dotenv import load_dotenv
from os import getenv
from collections import deque
from time import time

def get_vocab_from_batch(ds: Dataset, batch_size: int, offset: int):
    vocab = []
    for i in range(batch_size):
        if (offset + i) < ds.num_rows:
            text = ds[offset + i]["text"]
            words = [word.strip(string.punctuation) for word in text.split() if word.strip(string.punctuation).isalnum()]
            for word in words:
                if word not in vocab:
                    vocab.append(word)

    return (vocab, offset + batch_size)

def insert_item(q: deque, item: int):
        inserted = False
        for i in range(len(q)):
            if q[i] > item:
                q.insert(i, item)
                inserted = True
                break

        if not inserted:
            q.append(item)

def write_vocab(vocab: deque, min_done_idx: int):
    # assuming file is ran from project root dir
    with open("./vocab/vocab.txt", "w+") as file:
        file.write("\n".join(vocab))
    with open("./vocab/vocab_progress.txt", "w+") as file:
        file.write(str(min_done_idx))

def write_benchmarks(batch_size: int, n_proc: int, vocab_size: int, articles_processed: int, n_rows: int, time_info: float, avg_time_info: float):
    with open("./vocab/benchmarks.csv", "a+") as file:
        file.write(f"{batch_size}, {n_proc}, {vocab_size}, {articles_processed}/{n_rows}, {time_info:0.01f}, {avg_time_info:0.01f}\n")


if __name__ == "__main__":
    # had some problems using latest version, so I opted to use the already processed hf one
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    num_rows = ds.num_rows
    print(ds)

    load_dotenv()
    batch_size = int(getenv("BATCH_SIZE"))
    n_proc = int(getenv("N_PROC"))
    save_frq = 2 # will save the model after every n batches have been processed

    print("CONFIGURATION")
    print("---------------")
    print("batch_size: " + str(batch_size))
    print("n_proc: " + str(n_proc))
    print("---------------")

    vocab = deque(["SOS", "EOS", ",", ".", "\\n", "'", '"', "/", "+", "-", "[", "]", "{", "}", "(", ")", "_", "`", "?", "=", "&", "%", "#", "!", "@", "€", "$", ">", "<", "|", "*", "^", "§"])
    min_done_idx = 0
    done_idx_queue = deque()

    time_start = time()
    time_last = time_start
    
    with Pool(n_proc) as p:
        results = [
            p.apply_async(
                get_vocab_from_batch,
                (ds, batch_size, i * batch_size + min_done_idx)
            )
            for i in range(math.ceil((num_rows - min_done_idx)/batch_size))
        ]
        for i in range(len(results)):
            try:
                (parsed_vocab, done_idx) = results[i].get()
                for word in parsed_vocab:
                    if not word in vocab:
                        vocab.append(word)
                insert_item(done_idx_queue, done_idx)
                while True:
                    if len(done_idx_queue) > 0 and done_idx_queue[0] == min_done_idx + batch_size:
                        min_done_idx = done_idx_queue.popleft()
                    else:
                        break

                if i % save_frq == 1:
                    write_vocab(vocab, min_done_idx)
                    time_new = time()
                    write_benchmarks(batch_size, n_proc, len(vocab), min_done_idx, num_rows, (time_new-time_last) / save_frq, (time_new - time_start) / (i+1))
                    time_last = time_new

                if i == 100:
                    exit(0)

            finally:
                write_vocab(vocab, min_done_idx)

