import numpy as np
import pyhmmer
from Bio.Seq import Seq
from pyhmmer.easel import TextSequence
from tqdm import tqdm
from Bio.SeqRecord import SeqRecord
import math

def shannon_entropy(sequence):
    """
    Calculate the Shannon entropy of a sequence.

    Parameters:
    sequence (str or SeqRecord): The sequence to calculate the entropy of.

    Returns:
    float: The Shannon entropy of the sequence.
    """
    if isinstance(sequence, SeqRecord):
        sequence = str(sequence.seq)

    seq_list = list(sequence)
    unique_symbols = set(seq_list)
    M = float(len(seq_list))
    entropy_list = []
    for x in unique_symbols:
        n_i = seq_list.count(x)
        P_i = n_i / M
        entropy_i = P_i * (math.log(P_i, 2))
        entropy_list.append(entropy_i)

    sh_entropy = abs(sum(entropy_list))

    return sh_entropy

def compute_dielectric_constant(seq):
    """
    compute dielectric constant of a protein according to:
    Variations in Proteins Dielectric Constants
    Muhamed Amin and Jochen Küpper
    """

    amino_acid_volumes = {  # volume of the aminoacids
        'A': 81, 'C': 98, 'D': 102, 'E': 121, 'F': 160, 'G': 63,
        'H': 138, 'I': 141, 'K': 158, 'L': 139, 'M': 139, 'N': 112,
        'P': 109, 'Q': 132, 'R': 173, 'S': 92, 'T': 109, 'V': 119,
        'W': 193, 'Y': 168
    }

    amino_acids_electrons = {  # number of electrons of the aminoacids
        'A': 48, 'C': 49, 'D': 51, 'E': 57, 'F': 63, 'G': 40,
        'H': 61, 'I': 63, 'K': 58, 'L': 63, 'M': 62, 'N': 52,
        'P': 57, 'Q': 56, 'R': 65, 'S': 49, 'T': 53, 'V': 62,
        'W': 74,'Y': 63
    }


    amino_acid_alfa = {  # polarizability of the aminoacids
        'A': 8, 'C': 11, 'D': 12, 'E': 15, 'F': 18, 'G': 6,
        'H': 15, 'I': 14, 'K': 14, 'L': 13, 'M': 15, 'N': 11,
        'P': 11, 'Q': 13, 'R': 17, 'S': 9, 'T': 11, 'V': 12,
        'W': 23, 'Y': 19
    }


    # compute the sum of squared polarizability of the aminoacids
    total_tau = sum(  (amino_acids_electrons[aa]*amino_acid_alfa[aa]/4)**(1/2)
                      for aa in seq  )

    # compute the total number of electrons in the protein
    total_electrons = sum(amino_acids_electrons[aa] for aa in seq)

    # compute the total volume of the protein
    total_volume = sum(amino_acid_volumes[aa] for aa in seq)

    # compute protein average polarizability
    alpha =  (4/total_electrons)*(total_tau**2)

    # compute the polarizability term for clausius-mossotti relation
    cm_term = (4 * np.pi * alpha) / (3 * total_volume)

    # compute dielectrict constant using clausius-mossotti
    return (-1-cm_term * 2) / (cm_term -1)
def compute_tm_index(seq):
    weights = {

        'A': {'A': 100.0, 'C': 28.2, 'D': 100.0, 'E': 100.0, 'F': 100.0, 'G': 100.0, 'H': 100.0, 'I': 100.0,
              'K': 100.0, 'L': 100.0, 'M': 164.0, 'N': 100.0, 'P': 154.0, 'Q': 100.0, 'R': 62.1, 'S': 68.8,
              'T': 129.0, 'V': 100.0, 'W': 100.0, 'Y': 100.0},
        'C': {'A': 100.0, 'C': 100.0, 'D': 100.0, 'E': 26.7, 'F': 168.0, 'G': 100.0, 'H': 402.0, 'I': -43.0, 'K': 29.0,
              'L': 56.3, 'M': 100.0, 'N': -76.0, 'P': 159.0, 'Q': -8.5, 'R': 100.0, 'S': 100.0, 'T': 100.0,
              'V': 173.0, 'W': -153.0, 'Y': 100.0},
        'D': {'A': 48.6, 'C': -78.0, 'D': 100.0, 'E': 100.0, 'F': 100.0, 'G': 100.0, 'H': 93.0, 'I': 35.3,
              'K': 100.0, 'L': 100.0, 'M': 226.0, 'N': 100.0, 'P': 100.0, 'Q': 147.0, 'R': 140.0, 'S': 159.0,
              'T': 100.0, 'V': 100.0, 'W': 100.0, 'Y': 151.0},
        'E': {'A': 41.8, 'C': 29.3, 'D': 100.0, 'E': 110.0, 'F': 100.0, 'G': 174.0, 'H': 100.0, 'I': 100.0,
              'K': 100.0, 'L': 132.0, 'M': 23.4, 'N': 102.0, 'P': 100.0, 'Q': 156.0, 'R': 66.3, 'S': 68.6,
              'T': 106.0, 'V': 100.0, 'W': 25.0, 'Y': 140.0},
        'F': {'A': 100.0, 'C': 100.0, 'D': 142.0, 'E': 100.0, 'F': 3.45, 'G': 107.0, 'H': 177.0, 'I': -37.0,
              'K': 105.0, 'L': 100.0, 'M': 100.0, 'N': 100.0, 'P': 100.0, 'Q': 42.1, 'R': 100.0, 'S': 149.0,
              'T': 100.0, 'V': 100.0, 'W': 100.0, 'Y': 100.0},
        'G': {'A': 100.0, 'C': 161.0, 'D': -12.0, 'E': 178.0, 'F': 100.0, 'G': 124.0, 'H': 100.0, 'I': 136.0,
              'K': 100.0, 'L': 63.7, 'M': 153.0, 'N': 100.0, 'P': 61.0, 'Q': 132.0, 'R': 100.0, 'S': 100.0,
              'T': 100.0, 'V': 73.4, 'W': 151.0, 'Y': 11.7},
        'H': {'A': 100.0, 'C': 100.0, 'D': 24.7, 'E': 100.0, 'F': 248.0, 'G': 100.0, 'H': 100.0, 'I': 244.0,
              'K': 100.0, 'L': 100.0, 'M': 100.0, 'N': 100.0, 'P': 100.0, 'Q': 188.0, 'R': 25.9, 'S': 88.5,
              'T': 100.0, 'V': 24.5, 'W': 100.0, 'Y': 191.0},
        'I': {'A': 100.0, 'C': 32.1, 'D': 55.8, 'E': 100.0, 'F': 87.2, 'G': 150.0, 'H': -59.0, 'I': 100.0,
              'K': 100.0, 'L': 100.0, 'M': 100.0, 'N': 100.0, 'P': 100.0, 'Q': 60.1, 'R': 16.3, 'S': 100.0,
              'T': 100.0, 'V': 100.0, 'W': 211.0, 'Y': 100.0},
        'K': {'A': 126.0, 'C': 100.0, 'D': 100.0, 'E': 100.0, 'F': 153.0, 'G': 66.5, 'H': 24.5, 'I': 100.0,
              'K': 141.0, 'L': 100.0, 'M': 100.0, 'N': 3.49, 'P': 100.0, 'Q': 100.0, 'R': 137.0, 'S': 100.0,
              'T': 34.1, 'V': 174.0, 'W': 100.0, 'Y': 100.0},
        'L': {'A': 130.0, 'C': 100.0, 'D': 134.0, 'E': 64.4, 'F': 16.7, 'G': 105.0, 'H': 33.4, 'I': 100.0,
              'K': 47.5, 'L': 133.0, 'M': 100.0, 'N': 100.0, 'P': 116.0, 'Q': 95.1, 'R': 100.0, 'S': 136.0,
              'T': 100.0, 'V': 158.0, 'W': -69.0, 'Y': 100.0},
        'M': {'A': 150.0, 'C': 100.0, 'D': 25.1, 'E': 102.0, 'F': 100.0, 'G': 51.9, 'H': 100.0, 'I': 100.0,
              'K': 100.0, 'L': 32.2, 'M': -209.0, 'N': 164.0, 'P': 100.0, 'Q': 100.0, 'R': 100.0, 'S': 100.0,
              'T': 100.0, 'V': 100.0, 'W': 306.0, 'Y': 100.0},
        'N': {'A': 58.8, 'C': 100.0, 'D': 52.4, 'E': 143.0, 'F': 100.0, 'G': 138.0, 'H': 36.0, 'I': 100.0,
              'K': 165.0, 'L': 100.0, 'M': 219.0, 'N': -2.25, 'P': 100.0, 'Q': 100.0, 'R': 100.0, 'S': 100.0,
              'T': 100.0, 'V': 100.0, 'W': 100.0, 'Y': 88.1},
        'P': {'A': 100.0, 'C': 255.0, 'D': 100.0, 'E': 62.6, 'F': 3.45, 'G': 100.0, 'H': 189.0, 'I': 100.0,
              'K': 58.3, 'L': 100.0, 'M': 100.0, 'N': 100.0, 'P': 139.0, 'Q': 100.0, 'R': 159.0, 'S': 100.0,
              'T': 62.8, 'V': 43.6, 'W': -84.0, 'Y': 197.0},
        'Q': {'A': 36.0, 'C': 100.0, 'D': 165.0, 'E': 54.5, 'F': 168.0, 'G': -15.0, 'H': 100.0, 'I': 100.0,
              'K': 100.0, 'L': 54.2, 'M': 100.0, 'N': 116.0, 'P': 168.0, 'Q': 253.0, 'R': 47.5, 'S': 100.0,
              'T': 100.0, 'V': 100.0, 'W': 100.0, 'Y': 212.0},
        'R': {'A': 54.4, 'C': 100.0, 'D': 100.0, 'E': 142.0, 'F': 100.0, 'G': 61.3, 'H': 28.6, 'I': 100.0,
              'K': 137.0, 'L': 100.0, 'M': 100.0, 'N': 166.0, 'P': 34.0, 'Q': 15.8, 'R': 144.0, 'S': 100.0,
              'T': 100.0, 'V': 53.3, 'W': 158.0, 'Y': 100.0},
        'S': {'A': 100.0, 'C': 100.0, 'D': 143.0, 'E': 83.1, 'F': 144.0, 'G': 138.0, 'H': 184.0, 'I': 100.0,
              'K': 57.5, 'L': 100.0, 'M': 46.6, 'N': 100.0, 'P': 100.0, 'Q': 63.7, 'R': 67.6, 'S': 44.6,
              'T': 100.0, 'V': 100.0, 'W': 25.5, 'Y': 40.1},
        'T': {'A': 141.0, 'C': -62.1, 'D': 48.0, 'E': 115.0, 'F': 100.0, 'G': 100.0, 'H': 100.0, 'I': 158.0,
              'K': 100.0, 'L': 100.0, 'M': -37.2, 'N': 100.0, 'P': 52.3, 'Q': 100.0, 'R': 100.0, 'S': 100.0,
              'T': 100.0, 'V': 87.3, 'W': 100.0, 'Y': 100.0},
        'V': {'A': 168.0, 'C': 100.0, 'D': 140.0, 'E': 100.0, 'F': 26.3, 'G': 2.2, 'H': 100.0, 'I': 100.0,
              'K': 132.0, 'L': 100.0, 'M': 37.1, 'N': 161.0, 'P': 88.6, 'Q': 43.6, 'R': 93.6, 'S': 55.6,
              'T': 100.0, 'V': 146.0, 'W': 100.0, 'Y': 30.9},
        'W': {'A': 100.0, 'C': 100.0, 'D': -99.0, 'E': 100.0, 'F': 197.0, 'G': 151.0, 'H': 100.0, 'I': 100.0,
              'K': 100.0, 'L': 100.0, 'M': 100.0, 'N': 100.0, 'P': 100.0, 'Q': -40.6, 'R': 100.0, 'S': 100.0,
              'T': -26.8, 'V': 100.0, 'W': 100.0, 'Y': 100.0},
        'Y': {'A': 87.0, 'C': 100.0, 'D': 100.0, 'E': 63.5, 'F': 100.0, 'G': 67.3, 'H': 100.0, 'I': 153.0,
              'K': 121.0, 'L': 173.0, 'M': 100.0, 'N': 88.1, 'P': 100.0, 'Q': 149.0, 'R': 100.0, 'S': 100.0,
              'T': 100.0, 'V': 41.2, 'W': -23.0, 'Y': 21.6}}

    ti_tot = sum([weights[seq[i]][seq[i + 1]] for i in range(len(seq) - 1)])
    return ((100 / len(seq)) * ti_tot - 9372) / 398


def compute_contact_vector(cmap):
    return np.array([0. if k >= cmap.shape[0] else  # protein shorter than 400 aa
                     sum([cmap[i, i + k] for i in range(cmap.shape[0] - k)])  # count of the contact at distance k
                     for k in range(3, 400)
                     ])


def compute_distance(cvec1, cvec2):
    return np.abs(cvec1 - cvec2).sum() / np.sum(cvec1 + cvec2)


def compute_binned_distannce(cmap1, cmap2):
    return sum([
        compute_distance(
            compute_contact_vector(cmap1 > t),
            compute_contact_vector(cmap2 > t)
        )
        for t in np.linspace(.01, .99)
    ])


def load_pfam():
    with pyhmmer.plan7.HMMFile('lib/Pfam-A.hmm') as hmm_file:
        hmms = list(hmm_file)
    return hmms

def extract_profiles(seqs, hmms):

    tseqs = [  TextSequence(sequence=seq, name=seq.encode()).digitize(pyhmmer.easel.Alphabet.amino())
               for seq in tqdm(seqs,desc='parsing sequences', total=len(seqs)) ]

    print('finding domains')
    res = list(pyhmmer.hmmer.hmmscan(tseqs,hmms))

    return {  r.query.name.decode('ascii'):[    {   'desc':d.description,
                                    'start_idx':d.best_domain.env_from,
                                    'stop_idx':d.best_domain.env_to
                                }
                              for d in list(r.included)
                           ]
                         for r in tqdm(res,desc='parsing result',total=len(res))
            }

def distance_to_family(fmat, mu_indx):
    for i in range(fmat.shape[0]):
        if fmat[i,0] <= mu_indx <= fmat[i,1]:
            return 0
    return np.min(np.abs(mu_indx - fmat))


def entropy_around_mutation(seq, pos):

    start_idx, stop_idx = asymmetric_window(pos, len(seq), win_size=40)
    return shannon_entropy(Seq(seq[start_idx:stop_idx]))


def flexibility_around_mutation(flexi_wt, flexi_mu, pos):
    flexi_wt = np.array(flexi_wt)
    flexi_mu = np.array(flexi_mu)

    start_idx, stop_idx = asymmetric_window(pos, len(flexi_wt), win_size=40)
    flexi_wt = flexi_wt[start_idx:stop_idx].sum()

    start_idx, stop_idx = asymmetric_window(pos, len(flexi_mu), win_size=40)
    flexi_mu = flexi_mu[start_idx:stop_idx].sum()

    return flexi_wt - flexi_mu


def asymmetric_window(pos,seq_len, win_size):
    start_idx = pos- (win_size //2)
    stop_idx = pos + (win_size //2)
    if start_idx < 0: # not enogh aa before mutation, move stop index
        stop_idx  += np.abs(start_idx)
        start_idx = 0
    if stop_idx > seq_len:  # not enough aa after mutation, move start index ( if possible )
        start_idx = max((0, start_idx - (stop_idx - seq_len) ))
        stop_idx = seq_len
    return start_idx, stop_idx

def calculate_a3v(sequence):
    """Calculate the a3v value for each amino acid based on propensities."""
    aa_propensities = {
        'I': 1.822, 'F': 1.754, 'V': 1.594, 'L': 1.38, 'Y': 1.159, 'W': 1.037, 'M': 0.91,
        'C': 0.604, 'A': -0.036, 'T': -0.159, 'S': -0.294, 'P': -0.334, 'G': -0.535,
        'K': -0.931, 'H': -1.033, 'Q': -1.231, 'R': -1.24, 'N': -1.302, 'E': -1.412, 'D': -1.836 }
    return np.array([aa_propensities[aa] for aa in sequence])


def calculate_a4v(a3v_sequence):

    # assign window size according to sequence length
    if a3v_sequence.shape[0] <= 75:
        window_size = 5
    elif a3v_sequence.shape[0] <= 175:
        window_size = 7
    elif a3v_sequence.shape[0] <= 300:
        window_size = 9
    else:
        window_size = 11

    # add virtual residues to the sequence
    virtual_a3v = np.concatenate(([-1.625],a3v_sequence,[-1.085]))

    """Calculate the a4v using a sliding window."""
    ret= np.convolve(virtual_a3v, np.ones(window_size)/window_size, mode='valid')
    start = [ret[0]] * (window_size//2 - 1)
    end = [ret[-1]] * (window_size//2 - 1)
    return np.concatenate((start,ret,end))


def identify_hot_spots(a4v, sequence):
    """Identify Hot Spots in the sequence where a4v > HST and no proline is present."""
    hst = -0.02 # threshold precomputed from swissprot
    hot_spots = []
    start = None

    for i, (value, aa) in enumerate(zip(a4v, sequence)):
        if value > hst and aa != 'P':
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 5:  # Minimum 5 residues
                hot_spots.append((start, i-1))
            start = None
    return hot_spots

def compute_aggregation_profile(seq):
    """

    :param seq: string of the protein
    :return: numpy array of len(seq) size. i-th element is 0 if corresponding aa is not in a hotspot otherwise is equal
        to aminoacid a4v
    """
    a3v = calculate_a3v(seq)
    a4v = calculate_a4v(a3v)
    hotspots = identify_hot_spots(a4v, seq)
    a4v += 0.02 # center on aggregation threshold

    for i in range(len(seq)):
        if not any( [   start <= i <= stop for (start,stop) in hotspots] ):
            a4v[i]=0

    return a4v
