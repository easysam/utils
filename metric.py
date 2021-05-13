from sklearn.metrics import ndcg_score


def ndcg_at_k(_true_score, k=10):
    _len = len(_true_score)
    _true_score = _true_score < 10
    _score = [0] * _len
    _score[:10] = [1] * 10
    return ndcg_score([_true_score], [_score], k=k)