import streamlit as st
import openai
import base64
import datetime

st.set_page_config(page_title="サロンモデル化くん", page_icon="✂️", layout="centered")

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

NUM_PATTERNS = 3

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI APIキーが設定されていません。Streamlit Cloud の Secrets に OPENAI_API_KEY を追加してください。")
    st.stop()


def to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def analyze_image(img_bytes: bytes, instruction: str, max_tokens: int = 400) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{to_b64(img_bytes)}"}},
                {"type": "text", "text": instruction},
            ],
        }],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def analyze_hair(hair_bytes: bytes) -> str:
    return analyze_image(hair_bytes, (
        "Analyze the hairstyle in this image in extreme detail. Describe:\n"
        "- Bangs: shape, thickness, position, transparency, bundle structure\n"
        "- Part line: exact location (center/left/right)\n"
        "- Length: exact length (above ears / chin / shoulder / mid-back etc)\n"
        "- Layers and silhouette shape\n"
        "- Volume and fullness\n"
        "- Hair flow direction\n"
        "- Curl pattern and end shape (straight/inward curl/outward/wavy)\n"
        "- Left-right balance\n"
        "- EXACT hair color: describe root color, mid-shaft color, end color separately. "
        "Specify precise color (e.g. 'dark brown roots fading to warm caramel mid-shaft and honey blonde ends'). "
        "Note any highlights, gradient, or ombre.\n"
        "- Hair texture (smooth/silky/coarse/fluffy)\n"
        "Output only the description, no other text."
    ), max_tokens=600)


def analyze_face(face_bytes: bytes) -> str:
    return analyze_image(face_bytes, (
        "Analyze this person's facial features in detail. Describe:\n"
        "- Face shape (oval/round/square/heart etc)\n"
        "- Eye shape, size, color, and expression\n"
        "- Eyebrow shape and thickness\n"
        "- Nose shape\n"
        "- Lip shape and fullness\n"
        "- Skin tone\n"
        "- Overall facial impression and atmosphere\n"
        "- Approximate age range\n"
        "Output only the description, no other text."
    ), max_tokens=300)


def analyze_outfit(outfit_bytes: bytes) -> str:
    return analyze_image(outfit_bytes, (
        "Analyze the clothing in this image. Describe ONLY the clothing (ignore bags, accessories, jewelry):\n"
        "- Type of garment (top, dress, etc)\n"
        "- EXACT color — be very specific (e.g. 'vivid red', 'dusty rose', 'navy blue')\n"
        "- Fabric texture appearance\n"
        "- Neckline style\n"
        "- Sleeve style and length\n"
        "- Any patterns, prints, or details\n"
        "Output only the description, no other text."
    ), max_tokens=300)


def analyze_background(bg_bytes: bytes) -> str:
    return analyze_image(bg_bytes, (
        "Describe the background/setting in this image:\n"
        "- Setting type (indoor studio / outdoor / salon / etc)\n"
        "- Colors and tones\n"
        "- Lighting style\n"
        "- Any notable elements\n"
        "Output only the description, no other text."
    ), max_tokens=200)


def build_prompt(hair_desc: str, face_desc: str | None, outfit_desc: str | None, bg_desc: str | None) -> str:
    parts = [
        "Professional bust-up portrait photograph of a person. Ultra-sharp focus, maximum resolution, photorealistic.",
        "",
        f"=== HAIRSTYLE (most important — reproduce exactly) ===\n{hair_desc}",
        "",
    ]

    if face_desc:
        parts.append(f"=== FACE ===\n{face_desc}")
        parts.append("")

    if outfit_desc:
        parts.append(f"=== OUTFIT (clothing only, no bags or accessories) ===\n{outfit_desc}")
        parts.append("")

    if bg_desc:
        parts.append(f"=== BACKGROUND ===\n{bg_desc}")
        parts.append("")

    parts.append(
        "=== RULES ===\n"
        "- Hands completely empty, placed at sides of body or near thighs, NOT near face\n"
        "- Hair color must exactly match the hairstyle description — no color correction\n"
        "- Outfit color must exactly match the outfit description — no darkening or desaturation\n"
        "- Each color (hair/outfit/background) controlled independently\n"
        "- Edge sharpness maximum, no blur"
    )

    return "\n".join(parts)


def generate_image(prompt: str) -> bytes:
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1536",
    )
    return base64.b64decode(response.data[0].b64_json)


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
            status.info("ヘアスタイルを分析中...")
            hair_desc = analyze_hair(st.session_state.hair_img)
            progress.progress(10)

            face_desc = None
            if st.session_state.face_img:
                status.info("顔を分析中...")
                face_desc = analyze_face(st.session_state.face_img)
            progress.progress(20)

            outfit_desc = None
            if st.session_state.outfit_img:
                status.info("服装を分析中...")
                outfit_desc = analyze_outfit(st.session_state.outfit_img)
            progress.progress(30)

            bg_desc = None
            if st.session_state.bg_img:
                status.info("背景を分析中...")
                bg_desc = analyze_background(st.session_state.bg_img)
            progress.progress(40)

            prompt = build_prompt(hair_desc, face_desc, outfit_desc, bg_desc)

            for i in range(NUM_PATTERNS):
                status.info(f"パターン {i+1} / {NUM_PATTERNS} を生成中...（1〜2分かかります）")
                progress.progress(40 + int((i / NUM_PATTERNS) * 55))
                results.append(generate_image(prompt))

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
