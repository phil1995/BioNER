import fasttext
min_count = 10
context_size = 10
emb_size = 300
threads = 20
corpus_prefix = "pubmed"

fastText = "/data/scratch/schmidpp/fastText/fasttext"
train_data = "/data/scratch/schmidpp/PreprocessedPubMedAbstracts.txt"
output_dir = "/data/scratch/schmidpp/fasttext_embeddings/"


def generate_fasttext_embedding(min_n, max_n):
    model = fasttext.train_unsupervised(input=train_data,
                                        model='skipgram',
                                        lr=0.05,
                                        dim=emb_size,
                                        ws=context_size,
                                        neg=5,
                                        loss='ns',
                                        thread=threads,
                                        minCount=min_count,
                                        epoch=15,
                                        t=1e-5,
                                        minn=min_n,
                                        maxn=max_n)

    model.save_model(f'{output_dir}{corpus_prefix}.fasttext.{min_n}-{max_n}ngrams.neg5.1e-5_subs.bin')


if __name__ == '__main__':
    generate_fasttext_embedding(3, 3)
    generate_fasttext_embedding(3, 4)
    generate_fasttext_embedding(3, 5)
    generate_fasttext_embedding(3, 6)

    generate_fasttext_embedding(4, 4)
    generate_fasttext_embedding(4, 5)
    generate_fasttext_embedding(4, 6)

    generate_fasttext_embedding(5, 5)
    generate_fasttext_embedding(5, 6)

    generate_fasttext_embedding(6, 6)
