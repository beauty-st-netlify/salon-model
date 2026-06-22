import streamlit as st
import base64
import datetime
import json
import io
import requests
from google import genai
from google.genai import types

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

FOLDER_ID = "1JxCpIuHzIQZDjuQt5UG8KyOqdkbTYLPt"
NUM_PATTERNS = 3

try:
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Gemini APIキーが設定されていません。Streamlit Cloud の Secrets に GEMINI_API_KEY を追加してください。")
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


GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


def generate_with_images(
    hair_bytes: bytes,
    face_bytes: bytes | None,
    outfit_bytes: bytes | None,
    bg_bytes: bytes | None,
) -> bytes:
    # Gemini 2.5 Flash Image（通称 nano-banana）で複数参照を1枚に合成する。
    # 複数の参照画像を渡しての合成＋人物の同一性維持が得意なモデル。
    image_inputs = [
        ("画像1", hair_bytes, "ヘアスタイル参照。髪（形・髪型・髪色・長さ・前髪・分け目・毛流れ・毛先）だけをここから採用する。この画像の顔・人物は出力に一切使わない。"),
    ]
    idx = 2
    if face_bytes:
        image_inputs.append((f"画像{idx}", face_bytes, "顔参照。出力人物の顔・顔立ち・肌・同一性は必ずこの人物にする。この画像の髪・服・背景は使わない（顔だけ使う）。"))
        idx += 1
    if outfit_bytes:
        image_inputs.append((f"画像{idx}", outfit_bytes, "服装参照。服のみ採用（バッグ・小物・アクセサリーは除去）。服の色を完全維持する。"))
        idx += 1
    if bg_bytes:
        image_inputs.append((f"画像{idx}", bg_bytes, "背景参照。"))

    label_text = "\n".join(f"{name} = {desc}" for name, _, desc in image_inputs)
    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\nーーーーーーーーーーーーーー\n入力画像の割り当て\nーーーーーーーーーーーーーー\n"
        + label_text
        + "\n\n上記ルールに厳密に従い、髪は画像1から、顔は顔参照画像の人物で、1枚の人物画像を生成すること。出力は画像のみ。"
    )

    contents = [prompt]
    for _name, b, _desc in image_inputs:
        contents.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))

    # 縦長ポートレート（2:3）で生成。SDK 版差異に備えて config 無しにフォールバック。
    try:
        config = types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="2:3"),
        )
        resp = gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL, contents=contents, config=config,
        )
    except Exception:
        resp = gemini_client.models.generate_content(
            model=GEMINI_IMAGE_MODEL, contents=contents,
        )

    for cand in (resp.candidates or []):
        for part in (cand.content.parts or []):
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data
    raise Exception("画像が生成されませんでした。もう一度お試しください。")


def save_to_drive(image_bytes: bytes, filename: str) -> str | None:
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaInMemoryUpload
        from google.oauth2 import service_account

        creds_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = build("drive", "v3", credentials=creds)
        meta = {"name": filename, "parents": [FOLDER_ID]}
        media = MediaInMemoryUpload(image_bytes, mimetype="image/png")
        f = service.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
        return f.get("webViewLink")
    except Exception as e:
        st.warning(f"Drive保存をスキップしました: {e}")
        return None


# ---------- UI ----------

st.title("✂️ サロンモデル化くん")

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
        st.subheader(f"⚙️ {NUM_PATTERNS}パターン生成中...")

        progress = st.progress(0)
        status = st.empty()
        results = []

        try:
            for i in range(NUM_PATTERNS):
                status.info(f"パターン {i+1} / {NUM_PATTERNS} を生成中...（1〜2分かかります）")
                progress.progress(int((i / NUM_PATTERNS) * 90))

                img = generate_with_images(
                    st.session_state.hair_img,
                    st.session_state.face_img,
                    st.session_state.outfit_img,
                    st.session_state.bg_img,
                )
                results.append(img)

                # Drive保存
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_to_drive(img, f"サロンモデル_{ts}_パターン{i+1}.png")

            st.session_state.result_imgs = results
            progress.progress(100)
            st.rerun()

        except Exception as e:
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
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 同じ素材で再生成", use_container_width=True):
                st.session_state.result_imgs = None
                st.rerun()
        with col2:
            if st.button("🆕 最初からやり直す", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
