import os
import pickle
import re
import sys

from utils import duration, args, print, save_file

print('name:\n{}'.format(args.arg_name))

training_log = re.compile(r'^agent_training_log_\d{3}\.pickle$') 

os.chdir(save_file)
complete_order = args.arg_title[3:-3].split('+')
folders = [o for o in complete_order if o not in ['empty_space', 'break']]


def truncate_lists_in_dict(d):
    """Ensure inner lists are truncated to the shortest length across entries."""
    for key, value in d.items():
        if isinstance(value, list) and all(isinstance(item, list) for item in value):
            if value:
                lengths = [len(lst) for lst in value if lst]
                if lengths:
                    min_length = min(lengths)
                    d[key] = [lst[:min_length] for lst in value]
                else:
                    d[key] = [[] for _ in value]
    return d


# Iterate over folders with robots' saved dictionaries of data.
for folder in folders:
    training_logs = {}

    files = os.listdir(folder)
    files.sort()

    print('{} files in folder {}.'.format(len(files), folder))

    # Only process files full of information regarding plotting.
    filtered_files = [file for file in files if training_log.match(file)]

    if len(filtered_files) == 0:
        print(f'No matching files in {folder}, skipping.')
        continue

    # Process each file and populate the combined dictionaries.
    for file in filtered_files:
        print(file)

        with open(os.path.join(folder, file), 'rb') as handle:
            saved_log = pickle.load(handle)

        for key in saved_log.keys():
            if key not in training_logs:
                training_logs[key] = []
            d[key].append(saved_d[key])

    # Truncate lists in dictionaries.
    plot_dict = truncate_lists_in_dict(plot_dict)

    # Write the new output files (overwrite if they already exist).
    with open(os.path.join(folder, 'training_logs.pickle'), 'wb') as handle:
        pickle.dump(training_logs, handle)

    # Remove only the files we processed
    for file in filtered_files:
        os.remove(os.path.join(folder, file))

    print(f'Finished processing and saving for folder {folder}.')

print('\nDuration: {}. Done!\n'.format(duration()))
