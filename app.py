import streamlit as st
import openai
import base64
import datetime
import json
import requests

st.set_page_config(page_title="サロンモデル化くん", page_icon="✂️", layout="centered")

# ---------- セッション初期化 ----------
for key, val in [
    ("step", 0),
    ("hair_img", None),
    ("face_img", None),
    ("outfit_img", None),
    ("bg_img", None),
    ("result_imgs", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

FOLDER_ID = "1JxCpIuHzIQZDjuQt5UG8KyOqdkbTYLPt"
NUM_PATTERNS = 3

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


SYSTEM_INSTRUCTION = """You are an image synthesis AI. Generate ONE composite portrait image.

IMAGE ORDER AND ROLES:
- Image 1 (REQUIRED): Hairstyle reference — highest priority
- Image 2 (optional): Face reference — use for facial impression only (internal analysis only, do not output face)
- Image 3 (optional): Outfit reference — clothing only
- Image 4 (optional): Background reference

===== HAIRSTYLE (HIGHEST PRIORITY — TREAT AS FULLY LOCKED ELEMENT) =====
Extract and reproduce ALL of the following with EXACT fidelity:
- Bangs: bundle structure, position, thickness, gaps, transparency — CENTER PART FORBIDDEN
- Part line: exact position, do NOT move
- Length, layers, silhouette (width/height/volume): exact match
- Hair flow direction: exact match
- Curl pattern, ends: maintain inward curl exactly
- Left-right balance, face-framing hair: exact match

HAIRSTYLE IS AN IMMOVABLE COMPOSITE PART — FORBIDDEN:
- Any modification, correction, or regeneration of the hair
- Resizing, compressing, blurring, smoothing, redrawing, or regenerating
REQUIRED:
- Treat hair as the topmost front layer at all times
- Preserve: fine strands, strand boundaries, bang transparency, strand tip thinness, texture grain
- Hair resolution must equal or exceed the hairstyle reference image

===== HAIR COLOR (HIGHEST PRIORITY) =====
Hair color = pixel-exact match to hairstyle reference. FORBIDDEN ALL of the following:
- White balance correction
- Tone correction
- Color temperature correction
- Saturation correction
- Brightness correction
- Contrast correction
- Color matching
- Background color adaptation
- Any averaging or unification of color

HAIR COLOR DISTRIBUTION — treat as multi-zone, NOT single color:
- Root color: exact hue/saturation/brightness match
- Mid-shaft color: exact hue/saturation/brightness match
- End color: exact hue/saturation/brightness match
- Reproduce the full gradient from root to ends with complete accuracy
- No flattening, averaging, or single-color substitution

===== COLOR INDEPENDENCE RULE =====
Hair color, outfit color, and background color are INDEPENDENTLY controlled. NEVER blend or harmonize.
- Hair color follows ONLY the hairstyle reference
- Outfit color follows ONLY the outfit reference
- Background color follows ONLY the background reference
- FORBIDDEN: changing outfit color based on hair color or background color
- FORBIDDEN: changing hair color based on background color
- If outfit input is red → output MUST be red. No darkening, desaturation, or tone unification.

===== OUTFIT PROCESSING =====
Extract clothing ONLY. Completely delete:
- All bags, handbags, straps, small items, accessories, jewelry, decorations
Do NOT generate any bags or props.
Outfit color: exact match to reference — no darkening, desaturation, or harmonization.

===== POSE CONTROL =====
Hands MUST be completely empty — holding nothing.
Required hand placement: at sides of body, or near thighs — far from face.
FORBIDDEN hand zones: chin, cheeks, mouth, nose, eyes, ears, neck.
FORBIDDEN poses: hand on chin, touching cheeks, fingers near face, touching hair.

===== SCALE AND COMPOSITION =====
- Face size, head size, distance, position: exact match to hairstyle reference
- Aspect ratio: exact match to hairstyle reference
- Bust-up portrait ratio maintained
- Face position fixed

===== RESOLUTION AND QUALITY =====
- Edge sharpness: maximum
- Detail: maximum
- FORBIDDEN: blur, smoothing, softening anywhere

===== VERIFICATION — CHECK ALL BEFORE OUTPUT =====
If ANY of the following fails, regenerate until all pass:
□ Hairstyle intact — no modification
□ Bangs position and structure correct
□ Hair color exactly matches reference
□ Hair color gradient (root/mid/end) preserved
□ Outfit color exactly matches reference
□ Outfit design intact
□ No color correction applied anywhere
□ Resolution not reduced
□ Hands are NOT near face
□ No bags, accessories, or props generated

===== PRIORITY ORDER =====
1. Hairstyle — highest
2. Hair color — highest
3. Outfit — highest
4. Outfit color — highest
5. Background
6. Resolution
7. Composition

===== OUTPUT =====
Output the composite image ONLY. No text, no description, no explanation, no questions, no symbols."""


def generate_with_images(
    hair_bytes: bytes,
    face_bytes: bytes | None,
    outfit_bytes: bytes | None,
    bg_bytes: bytes | None,
) -> bytes:
    content = []

    content.append(img_content(hair_bytes))
    if face_bytes:
        content.append(img_content(face_bytes))
    if outfit_bytes:
        content.append(img_content(outfit_bytes))
    if bg_bytes:
        content.append(img_content(bg_bytes))

    labels = ["Image 1 = Hairstyle reference (MOST IMPORTANT — reproduce exactly)."]
    idx = 2
    if face_bytes:
        labels.append(f"Image {idx} = Face reference (internal use only).")
        idx += 1
    if outfit_bytes:
        labels.append(f"Image {idx} = Outfit reference (clothing only, no bags/accessories).")
        idx += 1
    if bg_bytes:
        labels.append(f"Image {idx} = Background reference.")

    content.append({"type": "input_text", "text": SYSTEM_INSTRUCTION + "\n\n" + " ".join(labels)})

    response = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
        tools=[{"type": "image_generation"}],
    )

    for item in response.output:
        if hasattr(item, "type") and item.type == "image_generation_call":
            return base64.b64decode(item.result)

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
