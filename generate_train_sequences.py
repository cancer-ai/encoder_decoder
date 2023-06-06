import random as rand
sequences = [
    "ATGAAAGTAACCGTTGTTGGAGCAGGTGCAGTTGGTGCAAGTTGCGCAGAATATATTGCA",
    "ATTAAAGATTTCGCATCTGAAGTTGTTTTGTTAGACATTAAAGAAGGTTATGCCGAAGGT",
]
new_sequences = []
train_pairs = []
base_pairs = ['A', 'T', 'C', 'G']
for seq in sequences: 
    mod_seq = []
    mod_train = []
    for i in range(0, len(seq)): 
        rand_base = rand.sample(base_pairs, 1)
        modified = seq[:i] + rand_base[0] + seq[i+1:]
        mod_seq.append(modified)
        mod_train.append((i, rand_base[0]))
    print(mod_seq)
    print(mod_train)
    new_sequences.append(mod_seq)
    train_pairs.append(mod_train)
print(new_sequences)
print(train_pairs)
        
        
    