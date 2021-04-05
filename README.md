# embedding_CNN
learn alignment matrix permutations 

runs in python3, tested with keras v2.4.3 and tensorflow v2.4.1 and sci-kit learn v0.22.2

to train run:
`python3 theta_data_siamese_example.py`

right now it runs for 8 epochs, and takes about an hr on my laptop

when it completes it should produce 3 dirs of saved models:
`8.epoch.test
8.epoch.test_base
8.epoch.test_head`

the one ending in `_base` is the part of the model that produces embeddings.

then to viz embeddings use the code in `get_embeddings_out.ipynb`
