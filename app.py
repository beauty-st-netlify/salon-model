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


def img_content(img_bytes: bytes) -> dict:
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{to_b64(img_bytes)}"}


SYSTEM_INSTRUCTION = """You are an image synthesis AI. Generate ONE composite portrait image based on the reference images provided.

IMAGE ORDER AND ROLES:
- Image 1 (REQUIRED): Hairstyle reference — reproduce this hairstyle with absolute fidelity
- Image 2 (if provided): Face reference — use this person's facial impression
- Image 3 (if provided): Outfit reference — use only the clothing, remove all bags/accessories
- Image 4 (if provided): Background reference — use this background

HAIRSTYLE RULES (highest priority):
- Reproduce the hairstyle EXACTLY: bangs, part line, length, layers, silhouette, volume, hair flow, curl, ends
- Hair color: reproduce the exact gradient from root to mid to ends — no color correction allowed
- Hair is a fixed composite element — do not modify or regenerate it
- Maintain maximum resolution and detail: fine strands, strand boundaries, transparency in bangs

COLOR RULES:
- Hair color, outfit color, and background color are independently controlled — do not blend or harmonize them
- Outfit color must exactly match the reference — no darkening, desaturation, or tone unification
- If outfit is red, output must be red

OUTFIT RULES:
- Extract clothing only — delete all bags, handbags, straps, accessories, and jewelry
- Do not generate any bags or props

POSE RULES:
- Hands must be completely empty — no holding anything
- Hands must be placed at sides of body or near thighs — NOT near face, chin, cheeks, ears, neck, or hair
- Forbidden poses: hand on chin, touching cheeks, fingers near face, touching hair

COMPOSITION:
- Bust-up portrait, face position fixed
- Ultra-sharp, maximum resolution, no blur or smoothing

OUTPUT: Generate the composite image only. No text."""


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
        labels.append(f"Image {idx} = Face reference.")
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
    if st.session_state.result_img is None:
        st.subheader("⚙️ 画像を生成しています...")

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("画像を解析・合成中...（1〜2分かかります）")
            progress.progress(30)

            result = generate_with_images(
                st.session_state.hair_img,
                st.session_state.face_img,
                st.session_state.outfit_img,
                st.session_state.bg_img,
            )

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
