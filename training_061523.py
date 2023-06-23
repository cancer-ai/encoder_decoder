filename =  "/project/mayocancerai/GenSLM/Embedding_output.txt"
with open(filename) as f:
    lines = f.readlines()
    embedding = []
    #all_embeddings = []
    for line in lines: 
        split = line.split(" ")
        print(split)
        embedding.append(split)
        print(embedding)
    print(embedding)
    #all_embeddings.append(embedding)
    