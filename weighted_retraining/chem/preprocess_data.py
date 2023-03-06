"""
Preprocesses the data to be quickly read in and processed by the JTNN
"""
import argparse
from pathlib import Path
import pickle
from tqdm.auto import tqdm

# My imports
from weighted_retraining.chem.chem_utils import rdkit_quiet
from weighted_retraining.chem.chem_data import tensorize

def save_data(all_data, start, end, num_per_file, save_dir):
    # For easiness, pad all data at the start so the indices
    # of data and all_data are the same
    all_data = [None] * start + all_data
    assert all_data[start] is not None

    # Save to appropriate files
    for i in range(start, end, num_per_file):
        file_name = f"tensors_{i:010d}-{i+num_per_file:010d}.pkl"
        with open(Path(save_dir) / file_name, "wb") as f:
            pickle.dump(all_data[i:i+num_per_file], f)

if __name__ == "__main__":
    rdkit_quiet()

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train_path", type=str, required=True)
    parser.add_argument("-s", "--start", type=int, default=0, help="starting index")
    parser.add_argument(
        "-e", "--end", type=int, default=None, help="ending index, default is last one"
    )
    parser.add_argument("-n", "--num_per_file", type=int, default=1000)
    parser.add_argument("-d", "--save_dir", type=str, required=True)
    args = parser.parse_args()

    # Read in training smiles
    with open(args.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    # Check whether ending directory exists
    if not Path(args.save_dir).is_dir():
        raise RuntimeError(f"{args.save_dir} does not exist!")

    # Deal with data processing fraction
    assert args.start < len(data)
    if args.end is None or args.end > len(data):
        args.end = len(data)
    assert args.start < args.end, "empty index list"

    # Process all data in the regime start-end
    print("Processing data!!!")
    all_data = []
    last_save = args.start
    for i, s in enumerate(tqdm(data[args.start : args.end], smoothing=0, dynamic_ncols=True)):
        all_data.append(tensorize(s))
        if i % 1000 == 0:
            print(f"saving files from {last_save} to {args.start + i}")
            save_data(all_data, last_save, args.start+i, args.num_per_file, args.save_dir)
            last_save = args.start + i

    
    print(f"saving files from {last_save} to {args.start + len(all_data)}")
    save_data(all_data, last_save, last_save+len(all_data), args.num_per_file, args.save_dir)
    last_save = args.start + len(all_data)

    print(last_save)
