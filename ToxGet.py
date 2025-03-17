import argparse
import numpy as np
from os import listdir, path, mkdir

ff_help = "Path to the fasta file, or folder of fasta files, for processing."
xp_help = "Path to the fasta file with the known sequences to compare against."
of_help = "Path to the output file or folder"
th_help = "Threshold used to determine if an unknown sequence matches the known sequence: Default is 0.7."
ll_help = "Minimum length of the sequence to be considered for comparison: Default is 16."
lu_help = "Maximum length of the sequence to be considered for comparsion: Default is 144."
mc_help = "Minimum number of cysteines for a sequence to be considered for comparison: Default is 4."

parser = argparse.ArgumentParser("ToxGet: A simple method to retrieve cysteine-motif toxins")
parser.add_argument("-ff", type=str, help=ff_help, required=True)
parser.add_argument("-xp", type=str, help=xp_help, required=True)
parser.add_argument("-of", type=str, default="toxget_out", help=of_help)
parser.add_argument("-th", type=float, default=0.7, help=th_help)
parser.add_argument("-ll", type=int, default=16, help=ll_help)
parser.add_argument("-lu", type=int, default=144, help=lu_help)
parser.add_argument("-mc", type=int, default=4, help=mc_help)
args = parser.parse_args()


def open_and_trim(path, lengths):
    """
    1st step of the pipeline: open the file and get all
        sequences inside a range of desired length 
    path: path to the file for prediction
    lengths: a tuple of 2 integers in ascending order
    """
    # step 0: open the file to sort through
    f = open(path, "r")

    # step 1: filter by length and get an array
    all_infos = []
    all_seqs = []
    info = "" # stores the current info line
    seq = "" # stores the current sequence line

    l = next(f, None)
    while l is not None:
        l = l.strip()
        if l == "":  pass # skip blank lines
        elif l[0] == '>':
            if not info: # if info is empty
                info = l.strip()
            elif info:
                good = min(lengths) <= len(seq) and len(seq) <= max(lengths)
                if good:
                    all_infos.append(info)
                    all_seqs.append(seq)
                info = l # set info to the new line
                seq = "" # reset sequences

        else:
            seq = seq + l.strip()
        l = next(f, None)

    # check the last sequence
    good = min(lengths) <= len(seq) and len(seq) <= max(lengths)
    if good:
        all_infos.append(info)
        all_seqs.append(seq)
    info = "" # reset info and seq
    seq = ""
    f.close() # close out the file reading from

    for s in all_seqs:
        assert len(s) <= lengths[1]
        assert len(s) >= lengths[0]

    # turn into a numpy array of [info, seqs]
    all_infos = np.asarray(all_infos, dtype=str)[:,np.newaxis]
    all_seqs = np.asarray(all_seqs, dtype=str)[:,np.newaxis]
    arr = np.concatenate([all_infos, all_seqs], axis=1)
    return arr


def min_cysteines(arr, min):
    def count_cysteines(seq):
        total = 0
        for c in seq:
            if c == "C": total +=1
        return total / len(seq), total
    keep_indices = []
    for i in range(arr.shape[0]):
        _, t = count_cysteines(arr[i,1])
        if t >= min:
            keep_indices.append(i)
    return arr[keep_indices]

# can make any shape as long as it is divisible by 4
def get_wavelet_matrices(size):
    # wavelet element values
    a = 0.06737176
    b = 0.09419511
    c = 0.40580489
    d = 0.56737176

    # wavelet filters
    alpha = np.asarray([-a,b,c,d,d,c,b,-a])[:,np.newaxis]
    beta = np.asarray([-b,a,d,c,-c,-d,-a,b])[:,np.newaxis]
    gamma = np.asarray([-b,-a,d,-c,-c,d,-a,-b])[:,np.newaxis]
    delta = np.asarray([-a,-b,c,-d,d,-c,b,a])[:,np.newaxis]

    filters =[alpha,beta,gamma,delta]
    row_list = []

    for filter in filters: # iterate over filters
        start = 0
        end = start + 8
        num_rows = size // 4
        for i in range(num_rows): # each filter gets 4 rows
            row = np.zeros([size,1]) # create a blank row
            # wrap the filter around the back to the start of the row if needed
            if end > size: 
                # assign the first 4 elements of the filter to the end of  the row           
                row[start:] += filter[:4] 
                # get the last 4 elements of the  filter and assign to the start  of the row
                row[0:4] += filter[4:]
            else: # otherwise just put the filter in  the necessary place
                row[start:end] += filter # add the filter values to the row
            start += 4 # increment spaces
            end += 4
            row_list.append(row)
    return np.squeeze(np.asarray(row_list))


def cysteine_embed(seq, length):
    if len(seq) > length:
        seq = seq[:length]
    v = []
    for i in range(length):
        if i < len(seq):
            if seq[i] == 'C':
                v.append(1)
            else:
                v.append(0)
        else:
            v.append(0)
    vector = np.asarray(v)
    return vector


def wavelet_transform(seqs, W):
    change = W.shape[0] // 4
    if seqs.shape[1] < W.shape[0]:
        dif = W.shape[0] - seqs.shape[1]
        zeros = np.zeros(shape=[seqs.shape[0],dif], dtype="float32")
        seqs = np.concatenate([seqs, zeros], axis=1)
    dot = np.dot(W, seqs.T)
    A = np.dot(W.T[:,0:change], dot[0:change,:])
    #D1 = np.dot(W.T[:,change:change*2], dot[change:change*2])
    #D2 = np.dot(W.T[:,change*2:change*3], dot[change*2:change*3])
    #D3 = np.dot(W.T[:,change*3:change*4], dot[change*3:change*4])
    return seqs - A.T # D1+D2+D3


def class_cross_correlation(a, b):
    norm_a = a - np.mean(a)
    norm_t = b - np.mean(b, axis=1, keepdims=True) 

    num = np.sum((norm_a * norm_t),axis=1)
    den = np.sum(np.square(norm_a)) * np.sum(np.square(norm_t),axis=1)
    out = num / np.sqrt(den+1e-5)
    return out


def get_predictions(X, Y, threshold):
    keep_indices = []
    for k in range(X.shape[0]):
        s = X[k]
        max_out = np.max(np.abs(class_cross_correlation(s, Y)))
        if max_out > threshold:
            keep_indices.append(k)
    return keep_indices


def pipeline(sequences, embedding, length, W):
    x = np.asarray([embedding(s,length) for s in sequences])
    return wavelet_transform(x, W)


# get the file or files for prediction
in_path = args.ff # get the paths to the multiple fasta files
assert path.exists(in_path), "File path is not found"

write_path = args.of
if not path.exists(write_path):
    mkdir(write_path)

if path.isdir(in_path): # if the path is a folder, do this
    fasta_files = listdir(in_path) # get the fasta files
    # join the folder to the fasta file name for the full path
    fasta_paths = [path.join(in_path, fn) for fn in fasta_files]

else: # otherwise just get the singular file
    fasta_files = [in_path]
    fasta_paths = [in_path]


# get the comparison sequences
compare_path = args.xp
assert path.exists(compare_path), "Cannot find the fasta file used for comparison"

# value check the other arguments
threshold = args.th
assert(threshold > 0.0 and threshold < 1.0), "Threshold needs to be between 0 and 1: "+str(threshold)+" was given."

min_len = args.ll
assert (min_len > 0), "The minimum sequence length must be > 0"

max_len = args.lu
assert (max_len > 0), "The maximum sequence length must be > 0"

assert (min_len != max_len), "The minimum and maximum sequences lengths must be different."

if max_len < min_len:
    sequence_lengths = (max_len, min_len)
else:
    sequence_lengths = (min_len, max_len)

required_cysteines = args.mc # no check here is required

# get the size of the wavelet matrix, should be the closest larger multiple of 4
# to the max sequence length
n = sequence_lengths[1]
if n % 4 == 0:
    wavelet_size = n
else:
     d = n % 4
     wavelet_size = n + (4-d)


Y = open_and_trim(compare_path, sequence_lengths)[:,-1]
W = get_wavelet_matrices(size=wavelet_size)
Y = pipeline(Y, cysteine_embed, sequence_lengths[1], W)


for index, file_path in enumerate(fasta_paths):
    file_name = fasta_files[index]
    print("Current file:",file_name)

    # open, remove by length, and remove by cysteine count
    arr = open_and_trim(file_path, sequence_lengths)
    arr = min_cysteines(arr, required_cysteines)
    print("Remainder after removal by length and cysteine count:",arr.shape[0])

    # perform crosscorr with known toxins
    seq = pipeline(arr[:,1], cysteine_embed, sequence_lengths[1], W)
    keep_indices = get_predictions(seq, Y, threshold)
    print("Remainder after crosscorrelation:", len(keep_indices))

    # write the remaining sequences
    file_name = file_name[:file_name.index(".")] + "_toxget.fasta"
    arr = arr[keep_indices]
    wp = path.join(write_path, file_name)
    write_file = open(wp,"w",encoding="utf8")
    for i in range(arr.shape[0]):
        write_file.write(arr[i,0]+"\n")
        write_file.write(arr[i,1]+"\n")
    write_file.close()
    print()


    
