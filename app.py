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
    ("result_img", None),
    ("drive_url", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

FOLDER_ID = "1JxCpIuHzIQZDjuQt5UG8KyOqdkbTYLPt"

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI APIキーが設定されていません。Streamlit Cloud の Secrets に OPENAI_API_KEY を追加してください。")
    st.stop()


# ---------- ユーティリティ ----------

def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def gpt4o_vision(img_bytes: bytes, prompt: str) -> str:
    b64 = to_b64(img_bytes)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return resp.choices[0].message.content


def analyze_hair(img_bytes: bytes) -> str:
    return gpt4o_vision(img_bytes, """
Analyze this hairstyle image in extreme detail (in English):
- Bang/fringe: length, structure, parting, spacing, transparency between strands
- Part line: exact position (center/left/right, distance from center)
- Overall length: where hair ends (chin / shoulder / collarbone / chest / waist)
- Layering: layered / one-length / graduated
- Silhouette: width, height, volume zones
- Hair flow direction (crown, sides, back)
- Curl/wave pattern: straight / wavy / curly, degree and location
- Hair ends: inward curl / outward curl / straight
- Volume: root, mid-length, ends
- Left-right symmetry
- Face-framing pieces
- Hair color GRADIENT — root color, mid-shaft color, end color (use exact color names: e.g. dark ash brown roots, medium caramel mid, honey blonde ends)
- Texture: glossy / matte / natural
Output as a detailed paragraph suitable for image generation.
""")


def analyze_face(img_bytes: bytes) -> str:
    return gpt4o_vision(img_bytes, """
Describe this person's facial features for image generation reference:
face shape, skin tone, eye shape and color, eyebrow shape, nose, lips, overall impression.
Keep it concise and specific.
""")


def analyze_outfit(img_bytes: bytes) -> str:
    return gpt4o_vision(img_bytes, """
Describe ONLY the clothing visible in this image. Ignore all bags, handbags, purses, accessories, and jewelry.
Include:
- Garment type (top, dress, jacket, blouse, etc.)
- Exact color (be very specific: cherry red, ivory white, navy blue, etc.)
- Fabric appearance and any pattern
- Neckline style
- Sleeve style and length
- Fit and silhouette
Do NOT mention bags, accessories, or jewelry.
""")


def analyze_background(img_bytes: bytes) -> str:
    return gpt4o_vision(img_bytes, """
Describe this background scene for use in image generation.
Include: location type, dominant colors, lighting quality and direction, atmosphere.
""")


def build_prompt(hair_desc: str, face_desc: str | None, outfit_desc: str | None, bg_desc: str | None) -> str:
    parts = [
        "Professional portrait photograph, bust-up composition, ultra-sharp, high resolution, photorealistic.",
        f"HAIRSTYLE (highest priority — reproduce exactly): {hair_desc}",
        "Hair color gradient must be reproduced exactly as described. Do NOT apply any color correction, white balance adjustment, or tone matching.",
        "Hair structure is fixed: bang position, part line, silhouette, volume, curl, and ends must match exactly.",
    ]

    if face_desc:
        parts.append(f"Person's facial features: {face_desc}")

    if outfit_desc:
        parts.append(f"OUTFIT (reproduce color exactly): {outfit_desc}")
        parts.append("Outfit color must not be altered. No bags, no handbags, no accessories.")
    else:
        parts.append("Wearing simple neutral-colored clothing.")

    if bg_desc:
        parts.append(f"Background: {bg_desc}")
    else:
        parts.append("Clean neutral studio background.")

    parts.extend([
        "Hands relaxed at the sides of the body — NOT touching the face, hair, chin, cheeks, or neck.",
        "No bags, no handbags, no accessories, no props.",
        "Edge sharpness maximum. No blur, no smoothing. Fine hair strands and strand boundaries clearly visible.",
    ])

    return " ".join(parts)


def generate_image(prompt: str) -> bytes:
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        n=1,
    )
    url = resp.data[0].url
    return requests.get(url, timeout=60).content


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

# ステップインジケーター
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
    if st.session_state.result_img is None:
        st.subheader("⚙️ 画像を生成しています...")

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("ヘアスタイルを解析中...")
            progress.progress(10)
            hair_desc = analyze_hair(st.session_state.hair_img)

            face_desc = None
            if st.session_state.face_img:
                status.info("顔画像を解析中...")
                progress.progress(25)
                face_desc = analyze_face(st.session_state.face_img)

            outfit_desc = None
            if st.session_state.outfit_img:
                status.info("服装を解析中...")
                progress.progress(40)
                outfit_desc = analyze_outfit(st.session_state.outfit_img)

            bg_desc = None
            if st.session_state.bg_img:
                status.info("背景を解析中...")
                progress.progress(55)
                bg_desc = analyze_background(st.session_state.bg_img)

            status.info("画像を生成中...（30秒ほどかかります）")
            progress.progress(65)
            prompt = build_prompt(hair_desc, face_desc, outfit_desc, bg_desc)
            result = generate_image(prompt)

            status.info("Google Drive に保存中...")
            progress.progress(90)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_url = save_to_drive(result, f"サロンモデル_{ts}.png")

            st.session_state.result_img = result
            st.session_state.drive_url = drive_url
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
        st.image(st.session_state.result_img, use_container_width=True)
        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ ダウンロード",
                data=st.session_state.result_img,
                file_name=f"salon_model_{ts}.png",
                mime="image/png",
                use_container_width=True,
            )
        with col2:
            if st.session_state.drive_url:
                st.link_button("📁 Driveで確認", st.session_state.drive_url, use_container_width=True)

        st.markdown("---")
        if st.button("🔄 もう一度作る", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
