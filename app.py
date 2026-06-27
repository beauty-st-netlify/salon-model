import streamlit as st
import openai
import base64
import datetime
import json
import io
import time
import requests
import concurrent.futures
import threading

st.set_page_config(page_title="サロンモデル化くん", page_icon="✂️", layout="centered")

# ---------- セッション初期化 ----------
for key, val in [
    ("step", 0),
    ("hair_img", None),
    ("face_img", None),
    ("outfit_img", None),
    ("bg_img", None),
    ("result_imgs", None),   # list of 3 images
]:
    if key not in st.session_state:
        st.session_state[key] = val

NUM_PATTERNS = 2

# テスト用の合計生成回数の上限（全ユーザー共通）。
# カウンタは GitHub リポジトリの専用ブランチ usage-data 上の usage_count.json に永続保存する
# （main とは別ブランチなので、書き込んでもアプリの再デプロイは発生しない）。
USAGE_LIMIT = 3
GITHUB_REPO = "beauty-st-netlify/salon-model"
USAGE_BRANCH = "usage-data"
USAGE_PATH = "usage_count.json"

# 顔の類似度による自動リトライ設定
MAX_FACE_RETRIES = 2          # 顔が一致しない時に作り直す最大回数（0で無効）
FACE_SIM_THRESHOLD = 0.42     # SFaceのコサイン類似度。これ未満=別人とみなして作り直す（高いほど厳しい。標準0.363）
_face_lock = threading.Lock() # 顔モデルを複数スレッドから安全に使うためのロック

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI APIキーが設定されていません。Streamlit Cloud の Secrets に OPENAI_API_KEY を追加してください。")
    st.stop()


# ---------- ユーティリティ ----------

def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def img_content(img_bytes: bytes) -> dict:
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{to_b64(img_bytes)}"}


SYSTEM_INSTRUCTION = """あなたは画像合成専用AIです。
ユーザーがアップロードした画像を参考に、それぞれの要素を統合し、画像を生成します。
最初にアップロードされた画像は必ずヘアスタイル画像として扱います。

ーーーーーーーーーーーーーー
要素ソースの分離（最重要）
ーーーーーーーーーーーーーー
出力人物の「顔・顔立ち・肌・人物の同一性」は Image2（顔画像）の人物を採用する。出力の顔は必ず Image2 の人物にする。
出力の「髪・髪型・髪色・長さ・毛流れ・前髪・分け目・毛先・すべての毛」は Image1（ヘアスタイル画像）のみから取得する。
最重要：Image2 に写っている髪は完全に無視し、出力に一切反映しない。Image2 からは顔だけを使い、髪・服・背景は使わない。
Image1 からは髪だけを基準とし、Image1 の顔立ちは出力に使わない（顔は Image2）。
ヘアスタイル画像(Image1)から最優先抽出：前髪 / 分け目 / 長さ / レイヤー構造 / シルエット / 毛流れ / カール / 毛先 / ボリューム / 左右バランス / 顔周りの毛 / 髪色 / 質感

ーーーーーーーーーーーーーー
色制御分離ルール（最重要）
ーーーーーーーーーーーーーー
髪色、服色、背景色はそれぞれ独立して制御すること
髪色：ヘアスタイル画像の色を完全一致で維持する
服色：服装画像の色を完全一致で維持する。他の要素の影響を受けて変更してはいけない。髪色や背景に合わせて色補正してはいけない
背景色：背景画像に従う
全体のカラーバランス調整は禁止。各要素の色は個別に維持すること

ーーーーーーーーーーーーーー
髪型固定（最重要）
ーーーーーーーーーーーーーー
髪型は編集不可領域として扱う。一切の変更・補正・再生成を禁止する。
髪は生成してはいけない。ヘアスタイル画像の髪をそのまま使用すること。
髪は合成パーツとして扱うこと。最前面レイヤーとして保持すること。

ーーーーーーーーーーーーーー
服色固定（最重要）
ーーーーーーーーーーーーーー
服の色は服画像の色を絶対に変更してはいけない。
禁止：黒化 / 彩度低下 / トーン統一 / 色の平均化 / 環境に合わせた色補正
赤い服が入力された場合は必ず赤で出力すること。

ーーーーーーーーーーーーーー
解像度・ディテール固定（最重要）
ーーーーーーーーーーーーーー
髪の解像度はヘアスタイル画像と同等以上で維持する。
禁止：リサイズ / 圧縮 / ぼかし / スムージング / 再描画 / 再生成
保持対象：細い毛 / 毛束境界 / 前髪透け感 / 毛先の細さ / 質感粒度

ーーーーーーーーーーーーーー
髪色補正完全禁止（最重要）
ーーーーーーーーーーーーーー
髪色に対して一切の補正処理を行ってはいけない。
禁止：ホワイトバランス補正 / トーン補正 / カラー補正 / 色温度補正 / 彩度補正 / 明度補正 / コントラスト補正 / カラーマッチング / 背景色への適応
髪色は入力画像のピクセル情報を基準として扱う。

ーーーーーーーーーーーーーー
髪色分布固定（最重要）
ーーーーーーーーーーーーーー
髪色は単一色として扱ってはいけない。
根元・中間・毛先 それぞれの色相・明度・彩度を完全一致させること。

ーーーーーーーーーーーーーー
グラデーション維持
ーーーーーーーーーーーーーー
根元から毛先の色変化を完全再現する。単色化禁止。平均化禁止。均一化禁止。

ーーーーーーーーーーーーーー
前髪固定
ーーーーーーーーーーーーーー
束構造・位置・太さ・隙間・透け感を完全一致。中央割れ禁止。

ーーーーーーーーーーーーーー
分け目固定
ーーーーーーーーーーーーーー
分け目位置完全一致。移動禁止。

ーーーーーーーーーーーーーー
シルエット固定
ーーーーーーーーーーーーーー
横幅・高さ・ボリューム完全一致。

ーーーーーーーーーーーーーー
毛流れ固定
ーーーーーーーーーーーーーー
方向完全一致。

ーーーーーーーーーーーーーー
毛先固定
ーーーーーーーーーーーーーー
内巻き維持。

ーーーーーーーーーーーーーー
服処理（最重要）
ーーーーーーーーーーーーーー
服のみ抽出。
削除：バッグ / カバン / ストラップ / 小物 / アクセサリー / 装飾品
生成禁止：バッグ / 小物

ーーーーーーーーーーーーーー
ポーズ制御（最重要）
ーーーーーーーーーーーーーー
手は完全に空にする。物を持つ動作は禁止。

ーーーーーーーーーーーーーー
顔周り手位置制限（最重要）
ーーーーーーーーーーーーーー
手を顔周りに配置してはいけない。
禁止エリア：顎 / 頬 / 口 / 鼻 / 目 / 耳 / 首
禁止ポーズ：顎に手を当てる / 頬に触れる / 顔に指を添える / 髪を触る

ーーーーーーーーーーーーーー
手の位置固定
ーーーーーーーーーーーーーー
手は以下に配置：体の横 / 腰の下 / 太もも付近。顔から十分離すこと。

ーーーーーーーーーーーーーー
スケール固定（最重要）
ーーーーーーーーーーーーーー
顔サイズ・頭サイズ・距離・位置 完全一致。

ーーーーーーーーーーーーーー
アスペクト比固定（最重要）
ーーーーーーーーーーーーーー
縦横比完全一致。

ーーーーーーーーーーーーーー
構図固定
ーーーーーーーーーーーーーー
バストアップ比率維持。顔位置固定。

ーーーーーーーーーーーーーー
画質最大化（最重要）
ーーーーーーーーーーーーーー
エッジシャープ。細部最大化。ぼかし禁止。

ーーーーーーーーーーーーーー
優先順位
ーーーーーーーーーーーーーー
髪型最優先 / 髪色最優先 / 服装最優先 / 服色最優先 / 背景次点 / 解像度次点 / 構図次点
髪色はヘアスタイル画像を基準に維持する。服色は服装画像を基準に維持する。背景色は背景画像を基準に維持する。
髪色、服色、背景色は互いに影響させない。髪色を理由に服色を変更してはいけない。背景色を理由に服色を変更してはいけない。

ーーーーーーーーーーーーーー
検証・再生成ルール（最重要）
ーーーーーーーーーーーーーー
以下を確認：髪型崩れ / 前髪ズレ / 髪色ズレ / 服色ズレ / 服装デザイン崩れ / 色補正 / 解像度低下 / 手が顔周りにある / バッグや小物
1つでも該当：無効 → 再生成。完全一致まで繰り返す。

ーーーーーーーーーーーーーー
出力制御
ーーーーーーーーーーーーーー
最終出力は画像のみ。テキスト禁止。説明禁止。補足禁止。質問禁止。記号禁止。"""


def normalize_for_api(img_bytes: bytes, max_side: int = 2048) -> bytes:
    """アップ画像を gpt-image edit API が確実に受け付ける形に正規化する。

    スマホ画像などで RGBA / CMYK / パレット等のモードや特殊形式だと
    「Invalid image file or mode」で弾かれるため、必ず RGB の PNG に変換し、
    長辺が大きすぎる場合は縮小する。失敗時は原本を返す。
    """
    try:
        from PIL import Image

        pil = Image.open(io.BytesIO(img_bytes))
        pil = pil.convert("RGB")
        w, h = pil.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            pil = pil.resize((max(int(w * scale), 1), max(int(h * scale), 1)))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes


def _img_file(name: str, img_bytes: bytes):
    """OpenAI Images API に渡す file タプル (filename, データ, MIME)。

    全画像を RGB PNG に正規化してから渡す（モード/形式不正での 400 を防ぐ）。
    """
    clean = normalize_for_api(img_bytes)
    png_name = name.rsplit(".", 1)[0] + ".png"
    bio = io.BytesIO(clean)
    bio.name = png_name
    return (png_name, bio, "image/png")


def blur_hairstyle_face(img_bytes: bytes) -> bytes:
    """ヘアスタイル画像の顔だけを強くぼかす。

    モデルが「ヘアスタイル画像の顔」をそのままコピーして顔がブレるのを防ぐ目的。
    髪（顔周りの毛を含む）はそのまま残すため、検出した顔 boxを少し内側に縮めてぼかす。
    顔検出に失敗した場合は原本をそのまま返す（無害なフォールバック）。
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageFilter

        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(pil)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return img_bytes

        # 一番大きい顔（＝メインの人物）を対象にする
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # 顔周りの毛・輪郭を残すため box を1割内側に縮める
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        box = (
            max(x + pad_x, 0),
            max(y + pad_y, 0),
            min(x + w - pad_x, pil.width),
            min(y + h - pad_y, pil.height),
        )
        region = pil.crop(box).filter(ImageFilter.GaussianBlur(radius=max(8, max(w, h) // 6)))
        pil.paste(region, box)

        out = io.BytesIO()
        pil.save(out, format="JPEG", quality=95)
        return out.getvalue()
    except Exception:
        # OpenCV未導入・検出失敗などは原本にフォールバック
        return img_bytes


@st.cache_resource(show_spinner=False)
def _load_face_models():
    """OpenCV の顔検出(YuNet)・顔認識(SFace)モデルを読み込む。

    モデルは OpenCV Zoo から初回のみダウンロードして一時領域にキャッシュする。
    追加の pip 依存は不要（opencv-python-headless に同梱の API を使う）。
    """
    import cv2
    import os
    import tempfile
    import urllib.request

    base = tempfile.gettempdir()
    det_path = os.path.join(base, "face_detection_yunet_2023mar.onnx")
    rec_path = os.path.join(base, "face_recognition_sface_2021dec.onnx")
    det_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    rec_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    if not os.path.exists(det_path):
        urllib.request.urlretrieve(det_url, det_path)
    if not os.path.exists(rec_path):
        urllib.request.urlretrieve(rec_url, rec_path)

    detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
    return detector, recognizer


def face_feature(img_bytes: bytes):
    """画像から最大の顔の特徴量ベクトルを取り出す。顔が無い/失敗時は None。"""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        detector, recognizer = _load_face_models()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])  # 一番大きい顔
        aligned = recognizer.alignCrop(bgr, face)
        return recognizer.feature(aligned)
    except Exception:
        return None


def face_similarity(feat_a, feat_b) -> float:
    """2つの顔特徴量のコサイン類似度（高いほど同一人物）。失敗時は -1。"""
    try:
        import cv2

        _, recognizer = _load_face_models()
        cosine_flag = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
        return float(recognizer.match(feat_a, feat_b, cosine_flag))
    except Exception:
        return -1.0


def generate_pattern(hair_bytes, face_bytes, outfit_bytes, bg_bytes, target_feat):
    """1パターン生成。顔が target_feat と一致しなければ上限まで作り直す。"""
    last = None
    for _attempt in range(MAX_FACE_RETRIES + 1):
        img = generate_with_images(hair_bytes, face_bytes, outfit_bytes, bg_bytes)
        last = img
        if target_feat is None:
            return img  # 顔参照なし or 顔特徴が取れない → 判定せず採用
        with _face_lock:
            feat = face_feature(img)
            sim = face_similarity(target_feat, feat) if feat is not None else 1.0
        if feat is None or sim >= FACE_SIM_THRESHOLD:
            return img  # 一致 or 判定不能 → 採用
    return last  # 上限まで作り直しても不一致なら最後の1枚を返す


def generate_with_images(
    hair_bytes: bytes,
    face_bytes: bytes | None,
    outfit_bytes: bytes | None,
    bg_bytes: bytes | None,
) -> bytes:
    # MyGPT と同じ経路：実画像そのものを gpt-image-1 の画像編集(edits)へ複数入力する。
    images = [_img_file("hairstyle.jpg", hair_bytes)]
    labels = ["Image 1 = Hairstyle reference — use ONLY its hair (style/color/shape/length). Do NOT use its face; the output face comes from the face reference."]
    idx = 2
    if face_bytes:
        images.append(_img_file("face.jpg", face_bytes))
        labels.append(f"Image {idx} = Face reference — the OUTPUT face/identity MUST be this person. Use ONLY the face; IGNORE this image's hair, clothing, and background.")
        idx += 1
    if outfit_bytes:
        images.append(_img_file("outfit.jpg", outfit_bytes))
        labels.append(f"Image {idx} = Outfit reference (clothing only, no bags/accessories).")
        idx += 1
    if bg_bytes:
        images.append(_img_file("background.jpg", bg_bytes))
        labels.append(f"Image {idx} = Background reference.")

    prompt = SYSTEM_INSTRUCTION + "\n\n" + " ".join(labels)

    # gpt-image-2 = ChatGPT Images 2.0 と同じ最新モデル。全入力を自動で高忠実度処理する
    # ため input_fidelity は指定不可（顔・細部の保持はデフォルトで有効）。
    result = client.images.edit(
        model="gpt-image-2",
        image=images,
        prompt=prompt,
        size="1024x1536",
        quality="medium",
    )

    b64 = result.data[0].b64_json
    if not b64:
        raise Exception("画像が生成されませんでした。もう一度お試しください。")
    return base64.b64decode(b64)


# ---------- テスト用の利用回数カウンタ（GitHub リポジトリに永続保存） ----------
# GitHub Contents API を使い、usage-data ブランチの usage_count.json を読み書きする。
# Secrets に GITHUB_TOKEN（このリポジトリへの contents 書き込み権限）が必要。

def _gh_headers():
    token = st.secrets["GITHUB_TOKEN"]
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }


def get_usage():
    """(count, sha) を返す。読めない時は (None, None) でフェイルオープン。"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{USAGE_PATH}?ref={USAGE_BRANCH}"
        r = requests.get(url, headers=_gh_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        content = base64.b64decode(data["content"]).decode()
        count = int(json.loads(content).get("count", 0))
        return count, data["sha"]
    except Exception:
        return None, None


def set_usage(count, sha):
    """カウンタを書き込み、新しい sha を返す。失敗時は None。"""
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{USAGE_PATH}"
        body = {
            "message": f"usage count -> {count}",
            "content": base64.b64encode(json.dumps({"count": count}).encode()).decode(),
            "branch": USAGE_BRANCH,
            "sha": sha,
        }
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=15)
        r.raise_for_status()
        return r.json()["content"]["sha"]
    except Exception:
        return None


# ---------- UI ----------

st.title("✂️ サロンモデル化くん")


@st.cache_data(ttl=15, show_spinner=False)
def _remaining_display():
    c, _ = get_usage()
    if c is None:
        return None
    return max(USAGE_LIMIT - c, 0)


_rem = _remaining_display()
if _rem is not None:
    st.caption(f"🧪 テスト残り生成回数: {_rem} / {USAGE_LIMIT}")

STEPS = ["①ヘアスタイル", "②顔", "③服装", "④背景", "⑤生成"]
step = st.session_state.step
cols = st.columns(5)
for i, (col, label) in enumerate(zip(cols, STEPS)):
    if i < step:
        col.markdown(f"<div style='text-align:center;color:#4CAF50'>✅<br><small>{label}</small></div>", unsafe_allow_html=True)
    elif i == step:
        col.markdown(f"<div style='text-align:center;color:#FF6B35;font-weight:bold'>▶<br><small>{label}</small></div>", unsafe_allow_html=True)
    else:
        col.markdown(f"<div style='text-align:center;color:#999'>○<br><small>{label}</small></div>", unsafe_allow_html=True)

st.markdown("---")


# ===== STEP 0: ヘアスタイル =====
if step == 0:
    st.subheader("💇 ヘアスタイル画像をアップロード")
    st.caption("完成イメージのヘアスタイルの写真（必須）")

    uploaded = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"], key="u_hair")
    if uploaded:
        st.image(uploaded, width=320)
        if st.button("次へ →", type="primary", use_container_width=True):
            st.session_state.hair_img = uploaded.read()
            st.session_state.step = 1
            st.rerun()


# ===== STEP 1: 顔 =====
elif step == 1:
    st.subheader("👤 顔画像をアップロード")
    st.caption("顔の雰囲気・印象の参考として使用します（省略可）")

    uploaded = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"], key="u_face")
    if uploaded:
        st.image(uploaded, width=320)

    col1, col2 = st.columns(2)
    with col1:
        if uploaded and st.button("次へ →", type="primary", use_container_width=True):
            st.session_state.face_img = uploaded.read()
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("スキップ", use_container_width=True):
            st.session_state.face_img = None
            st.session_state.step = 2
            st.rerun()


# ===== STEP 2: 服装 =====
elif step == 2:
    st.subheader("👗 服装画像をアップロード")
    st.caption("着用させたい服装の写真（バッグ・アクセサリーは自動除去）")

    uploaded = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"], key="u_outfit")
    if uploaded:
        st.image(uploaded, width=320)

    col1, col2 = st.columns(2)
    with col1:
        if uploaded and st.button("次へ →", type="primary", use_container_width=True):
            st.session_state.outfit_img = uploaded.read()
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("スキップ", use_container_width=True):
            st.session_state.outfit_img = None
            st.session_state.step = 3
            st.rerun()


# ===== STEP 3: 背景 =====
elif step == 3:
    st.subheader("🏞️ 背景画像をアップロード")
    st.caption("背景として使いたい画像（省略可）")

    uploaded = st.file_uploader("画像を選択してください", type=["jpg", "jpeg", "png"], key="u_bg")
    if uploaded:
        st.image(uploaded, width=320)

    col1, col2 = st.columns(2)
    with col1:
        if uploaded and st.button("生成する →", type="primary", use_container_width=True):
            st.session_state.bg_img = uploaded.read()
            st.session_state.step = 4
            st.rerun()
    with col2:
        if st.button("スキップして生成", use_container_width=True):
            st.session_state.bg_img = None
            st.session_state.step = 4
            st.rerun()


# ===== STEP 4: 生成 =====
elif step == 4:
    if st.session_state.result_imgs is None:
        # テスト用の合計回数チェック（全ユーザー共通・GitHub永続）
        usage_count, usage_sha = get_usage()
        if usage_count is not None and usage_count >= USAGE_LIMIT:
            st.error(f"🚫 テスト用の生成回数の上限（合計 {USAGE_LIMIT} 回）に達しました。ご利用ありがとうございました。")
            st.stop()

        st.subheader(f"⚙️ {NUM_PATTERNS}パターンを同時生成中...")

        progress = st.progress(0)
        status = st.empty()
        status.info(f"{NUM_PATTERNS}パターンを並列生成中...（目安1〜2分・できた順に下へ表示します）")

        # できた順に表示するためのプレビュー枠
        preview_cols = st.columns(NUM_PATTERNS)
        placeholders = [c.empty() for c in preview_cols]
        start_time = time.time()

        # この生成を1回ぶん消費（多重起動・リロードでの超過を防ぐため生成前に記録）
        if usage_count is not None:
            usage_sha = set_usage(usage_count + 1, usage_sha)

        try:
            # 顔画像がある時は、ヘアスタイル画像の顔をぼかしてから渡す
            # （モデルがヘア画像の顔をコピーして顔がブレるのを防ぐ）。1回だけ処理。
            hair_for_gen = st.session_state.hair_img
            target_feat = None
            if st.session_state.face_img:
                hair_for_gen = blur_hairstyle_face(st.session_state.hair_img)
                # アップ顔の特徴量を1回だけ算出（メインスレッドでモデルを温める）
                target_feat = face_feature(st.session_state.face_img)

            args = (
                hair_for_gen,
                st.session_state.face_img,
                st.session_state.outfit_img,
                st.session_state.bg_img,
                target_feat,
            )

            # 3パターンを同時並行で生成（順次だと枚数分の時間がかかるため）。
            # 各パターンは顔が一致するまで自動リトライ（generate_pattern内）。
            results = [None] * NUM_PATTERNS
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PATTERNS) as ex:
                future_to_idx = {ex.submit(generate_pattern, *args): i for i in range(NUM_PATTERNS)}
                done = 0
                for fut in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    results[i] = fut.result()
                    done += 1
                    # できた順にその場で表示
                    placeholders[i].image(results[i], caption=f"パターン{i+1}", use_container_width=True)
                    progress.progress(int((done / NUM_PATTERNS) * 100))
                    elapsed = int(time.time() - start_time)
                    status.info(f"{done} / {NUM_PATTERNS} パターン完了（経過 {elapsed} 秒）")

            st.session_state.result_imgs = results
            st.rerun()

        except Exception as e:
            # 生成失敗時は消費した1回ぶんを戻す
            if usage_count is not None and usage_sha is not None:
                set_usage(usage_count, usage_sha)
            st.error(f"エラーが発生しました: {e}")
            if st.button("最初からやり直す"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    else:
        st.subheader("✅ 生成完了！")

        ts_base = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, img in enumerate(st.session_state.result_imgs):
            st.markdown(f"**パターン {i+1}**")
            st.image(img, use_container_width=True)
            st.download_button(
                f"⬇️ パターン{i+1} をダウンロード",
                data=img,
                file_name=f"salon_model_{ts_base}_{i+1}.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_{i}",
            )
            st.markdown("")

        st.markdown("---")
        st.markdown("### ♻️ 素材を差し替えて次を作成")
        st.caption("変えたい画像だけ差し替えてください。差し替えないものは前回のまま使います（例：顔・服・背景は固定でヘアだけ変える）。")

        # (session_stateキー, ラベル, 必須かどうか)
        slots = [
            ("hair_img", "💇 ヘアスタイル", True),
            ("face_img", "👤 顔", False),
            ("outfit_img", "👗 服装", False),
            ("bg_img", "🏞️ 背景", False),
        ]
        edit_cols = st.columns(4)
        for (key, label, required), col in zip(slots, edit_cols):
            with col:
                st.markdown(f"**{label}**" if required else f"**{label}**<br><small>（省略可）</small>", unsafe_allow_html=True)
                cur = st.session_state.get(key)
                if cur:
                    st.image(cur, use_container_width=True)
                else:
                    st.caption("（なし）")
                up = st.file_uploader(
                    "差し替え", type=["jpg", "jpeg", "png"],
                    key=f"re_{key}", label_visibility="collapsed",
                )
                if up is not None:
                    st.image(up, caption="↑ 差し替え後", use_container_width=True)
                # 任意素材は「なしにする」も可能に
                if not required and cur is not None:
                    if st.button("削除", key=f"del_{key}", use_container_width=True):
                        st.session_state[key] = None
                        st.rerun()

        st.markdown("")
        gen_col1, gen_col2 = st.columns(2)
        with gen_col1:
            if st.button("✨ この内容で生成", type="primary", use_container_width=True):
                # 差し替えがあったスロットだけ session に反映（getvalueは非破壊で複数回押しても安全）
                for key, _label, _required in slots:
                    up = st.session_state.get(f"re_{key}")
                    if up is not None:
                        st.session_state[key] = up.getvalue()
                st.session_state.result_imgs = None
                st.rerun()
        with gen_col2:
            if st.button("🆕 最初からやり直す", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
