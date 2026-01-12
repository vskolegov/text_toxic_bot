from pathlib import Path
import re
import pickle
import joblib
import json
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).parent
CLASSIC_DIR = BASE / 'models' / 'classic'
BILSTM_DIR = BASE / 'models' / 'bilstm'


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Minimal BiLSTM from notebook
class CorrectBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim=64):
        super(CorrectBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, -3.0)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(last_hidden)
        return output


# Simple lazy loader/cache
_cache = {}


def _load_classic_assets():
    if 'tfidf' in _cache:
        return
    tfidf = joblib.load(CLASSIC_DIR / 'tfidf_vectorizer.pkl')
    _cache['tfidf'] = tfidf


def _load_classic_model(name: str):
    mapping = {
        'logreg': 'model_logistic_regression.pkl',
        'svm': 'model_svm.pkl',
        'voting': 'model_voting_ensemble.pkl',
    }
    fname = mapping.get(name)
    if fname is None:
        raise ValueError('unknown classic model')
    if fname in _cache:
        return _cache[fname]
    mdl = joblib.load(CLASSIC_DIR / fname)
    _cache[fname] = mdl
    return mdl


def _load_bilstm():
    if 'bilstm' in _cache:
        return _cache['bilstm']
    w2i = pickle.load(open(BILSTM_DIR / 'word2idx.pkl', 'rb'))
    cfg = json.load(open(BILSTM_DIR / 'bilstm_config.json', 'r', encoding='utf-8'))
    ckpt = torch.load(BILSTM_DIR / 'correct_bilstm.pth', map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    # infer embedding dim
    emb_key = None
    for k in state_dict.keys():
        if 'embedding.weight' in k:
            emb_key = k
            break
    if emb_key is not None:
        emb_shape = state_dict[emb_key].shape
        emb_dim = emb_shape[1]
    else:
        emb_dim = 300

    embedding_matrix = np.zeros((len(w2i), emb_dim), dtype=np.float32)
    model = CorrectBiLSTM(vocab_size=len(w2i), embedding_dim=emb_dim, embedding_matrix=embedding_matrix)
    # load with non-strict to be tolerant
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _cache['bilstm'] = {'model': model, 'w2i': w2i, 'cfg': cfg}
    return _cache['bilstm']


def predict_toxicity(text: str, model_type: str = 'voting') -> dict:
    """Predict toxicity.

    model_type: one of 'logreg','svm','voting','bilstm'
    Returns dict with keys: 'label' (0/1), 'prob' (float), 'model'
    """
    if model_type in ('logreg', 'svm', 'voting'):
        _load_classic_assets()
        tfidf = _cache['tfidf']
        X = tfidf.transform([preprocess_text(text)])
        mdl = _load_classic_model(model_type)
        try:
            prob = float(mdl.predict_proba(X)[0, 1])
        except Exception:
            # fallback: if no predict_proba, use decision_function
            try:
                score = mdl.decision_function(X)[0]
                prob = 1.0 / (1.0 + np.exp(-score))
            except Exception:
                prob = float(mdl.predict(X)[0])
        label = int(prob >= 0.5)
        return {'label': label, 'prob': prob, 'model': model_type}

    if model_type == 'bilstm':
        data = _load_bilstm()
        model = data['model']
        w2i = data['w2i']
        cfg = data['cfg']
        max_len = cfg.get('max_len', 100)
        txt = preprocess_text(text)
        words = txt.split()[:max_len]
        idxs = [w2i.get(w, 1) for w in words]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))
        X = torch.LongTensor([idxs])
        with torch.no_grad():
            logits = model(X)
            prob = float(torch.sigmoid(logits).item())
            label = int(prob >= 0.5)
        return {'label': label, 'prob': prob, 'model': 'bilstm'}

    raise ValueError('unknown model_type')


if __name__ == '__main__':
    samples = [
        'Ты ничтожество, никто тебя не слушает.',
        'Иди вон, твои идеи — отстой.',
        'Закрой рот, ты ничего не понимаешь.'
    ]
    for m in ('logreg', 'svm', 'voting', 'bilstm'):
        print('---', m)
        for s in samples:
            r = predict_toxicity(s, m)
            print(s[:40], '->', r)
