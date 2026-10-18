from pathlib import Path
import streamlit as st

st.set_page_config(page_title='차량 소리 판정', page_icon='🚗')
st.title('🚗 차량 소리 판정')
st.write('녹음하거나 WAV 파일을 올린 뒤 분석 버튼을 누르세요.')
st.caption('녹음 완료 후 앞부분 최대 10초를 분석합니다. 연속 실시간 감지는 아닙니다.')
BASE = Path(__file__).resolve().parent
MODEL = BASE / 'vehicle_classifier.joblib'

@st.cache_resource(show_spinner=False)
def get_bundle(signature):
    from inference import load_bundle
    return load_bundle(MODEL)

@st.cache_resource(show_spinner=False)
def get_yamnet():
    from inference import load_yamnet
    return load_yamnet()

if not MODEL.is_file():
    st.info('Colab에서 받은 vehicle_classifier.joblib을 app.py와 같은 폴더에 넣어주세요.')
    st.stop()

source = st.radio('입력 방식', ['마이크 녹음', 'WAV 업로드'], horizontal=True)
if source == '마이크 녹음':
    audio = st.audio_input('주변 소리를 녹음하세요', sample_rate=16000)
else:
    audio = st.file_uploader('WAV 파일 선택', type=['wav'])

with st.expander('판정 기준 조절'):
    override = st.checkbox('저장된 기준 대신 직접 설정')
    threshold = st.slider('차량 판정 임계값', 0.01, 0.99, 0.50, 0.01,
                          disabled=not override)
    st.caption('기준을 낮추면 차량을 더 많이 감지하지만 오탐도 늘어날 수 있습니다.')

if audio is not None:
    raw = audio.getvalue()
    st.audio(raw, format='audio/wav')
    if st.button('차량 소리 분석', type='primary'):
        if len(raw) > 25 * 1024 * 1024:
            st.error('25MB 이하 WAV 파일을 사용하세요.')
            st.stop()
        try:
            with st.spinner('모델을 준비하고 분석 중입니다. 첫 실행에는 YAMNet 다운로드가 필요합니다.'):
                stat = MODEL.stat()
                bundle = get_bundle((stat.st_mtime_ns, stat.st_size))
                yamnet = get_yamnet()
                from inference import analyze
                score, info = analyze(raw, bundle, yamnet)
            if score is None:
                st.warning('입력이 무음에 가깝습니다. 마이크와 녹음 내용을 확인하세요.')
            else:
                limit = threshold if override else float(bundle['threshold'])
                if score >= limit:
                    st.warning('🚗 차량 소리로 판정했습니다.')
                else:
                    st.success('비차량 소리로 판정했습니다.')
                st.metric('차량 점수', f'{score:.3f}')
                st.progress(score)
                st.caption(f'판정 기준 {limit:.2f} · 분석 구간 {info["used_seconds"]:.1f}초')
                if info['cropped']:
                    st.info('긴 파일이므로 앞부분 10초만 분석했습니다.')
                if info['padded']:
                    st.info('1초보다 짧아 나머지를 0으로 채워 분석했습니다.')
        except Exception as exc:
            st.error(f'분석하지 못했습니다: {exc}')
            with st.expander('오류 상세'):
                st.exception(exc)

st.caption('차량 소리 유무를 시험하는 앱입니다. 점수는 거리나 충돌 위험도가 아닙니다.')
