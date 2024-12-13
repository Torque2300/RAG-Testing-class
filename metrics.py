from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_f1(predicted, reference, lang='ru'):
    _, _, f1 = score([predicted], [reference], lang=lang, verbose=False)
    return f1.mean().item()


def calculate_recall(predicted, reference, lang='ru'):
    _, r, _ = score([predicted], [reference], lang=lang, verbose=False)
    return r.mean().item()


def calculate_precision(predicted, reference, lang='ru'):
    p, _, _ = score([predicted], [reference], lang=lang, verbose=False)
    return p.mean().item()


def calculate_cosine_similarity(predicted, reference, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
    sentence_transformer = SentenceTransformer(model_name)
    embeddings = sentence_transformer.encode([predicted, reference])
    return cosine_similarity([embeddings[0]], [embeddings[1]])
