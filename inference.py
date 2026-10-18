"""Matches the Colab YAMNet + StandardScaler + LogisticRegression pipeline."""
import io
import warnings
import numpy as np
import soundfile as sf
import librosa

YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'

def load_bundle(path):
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            bundle = joblib.load(path)
    except InconsistentVersionWarning as exc:
        raise ValueError(
            f'학습과 scikit-learn 버전이 다릅니다. requirements.txt의 scikit-learn 줄을 '
            f'scikit-learn=={exc.original_sklearn_version} 로 바꾼 뒤 재배포하세요.'
        ) from exc
    if not isinstance(bundle, dict) or 'classifier' not in bundle:
        raise ValueError('Colab ⑦에서 저장한 vehicle_classifier.joblib이 필요합니다.')
    expected = {'sample_rate':16000, 'max_seconds':10, 'min_seconds':1,
                'pooling':'mean', 'yamnet_url':YAMNET_URL}
    for key, value in expected.items():
        if bundle.get(key) != value:
            raise ValueError(f'지원하지 않는 모델 설정: {key}. 제공한 Colab 코드로 저장한 모델을 사용하세요.')
    model = bundle['classifier']
    if set(model.classes_) != {0, 1} or model.n_features_in_ != 1024:
        raise ValueError('1,024차원 특징을 받는 차량(1)/비차량(0) 분류기가 필요합니다.')
    threshold = float(bundle.get('threshold', .5))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('모델 임계값이 올바르지 않습니다.')
    return bundle

def load_yamnet():
    import tensorflow_hub as hub
    return hub.load(YAMNET_URL)

def prepare_audio(raw):
    with sf.SoundFile(io.BytesIO(raw)) as f:
        sr = f.samplerate
        duration = len(f) / sr
        audio = f.read(frames=int(sr * 10), dtype='float32', always_2d=True)
    if not len(audio):
        raise ValueError('빈 오디오입니다. 다시 녹음하세요.')
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError('오디오에 잘못된 값이 있습니다.')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000-len(audio)))
    audio = np.clip(audio, -1, 1).astype(np.float32)
    return audio, {'original_seconds':duration, 'used_seconds':min(duration, 10),
                   'padded':duration < 1, 'cropped':duration > 10,
                   'silent':bool(np.max(np.abs(audio)) < 1e-6)}

def analyze(raw, bundle, yamnet):
    audio, info = prepare_audio(raw)
    if info['silent']:
        return None, info
    _, embeddings, _ = yamnet(audio)
    feature = embeddings.numpy().mean(axis=0).reshape(1, -1)
    if feature.shape != (1, 1024) or not np.isfinite(feature).all():
        raise ValueError('YAMNet 특징 추출에 실패했습니다.')
    classifier = bundle['classifier']
    column = list(classifier.classes_).index(1)
    score = float(classifier.predict_proba(feature)[0, column])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('모델 출력이 올바르지 않습니다.')
    return score, info
