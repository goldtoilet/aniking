import os
import io
import base64

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from PIL import Image
import numpy as np

# imageio가 설치되지 않은 환경에서도 앱이 죽지 않게 예외 처리
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

# =========================
# .env 로 환경변수 로드 (로컬 개발용)
# =========================
load_dotenv()

# =========================
# 페이지 기본 설정 & 스타일
# =========================
st.set_page_config(
    page_title="imageking",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    textarea {
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .logo-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        background: #F3F4FF;
        color: #444;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .logo-badge span.emoji {
        font-size: 1rem;
    }
    .small-text-cell {
        font-size: 0.8rem;
        line-height: 1.3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 환경변수 가져오기
# =========================
def get_env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return value if value is not None else default


GPT_API_KEY = get_env("GPT_API_KEY", "")

if not GPT_API_KEY:
    st.error("❌ GPT_API_KEY 가 설정되어 있지 않습니다. .env 또는 환경변수를 확인해주세요.")
    st.stop()

client = OpenAI(api_key=GPT_API_KEY)

# =========================
# 이미지 / 영상 모델 프리셋
# =========================
IMAGE_MODELS = {
    "OpenAI gpt-image-1": "gpt-image-1",
}

VIDEO_MODELS = {
    "이미지 1장 → MP4 (로컬 합성)": "local_single_image_mp4",
}

# =========================
# 세션 상태 기본값
# =========================
st.session_state.setdefault("prompt_text", "")
st.session_state.setdefault("image_b64", None)
st.session_state.setdefault("image_model_label", "OpenAI gpt-image-1")
st.session_state.setdefault("image_orientation", "정사각형 1:1 (1024x1024)")
st.session_state.setdefault("image_quality", "low")

st.session_state.setdefault("video_model_label", "이미지 1장 → MP4 (로컬 합성)")
st.session_state.setdefault("seconds_per_scene", 3.0)
st.session_state.setdefault("video_bytes", None)
st.session_state.setdefault("video_error_msg", None)

# =========================
# 유틸 함수들
# =========================
def get_image_params():
    orientation = st.session_state.get("image_orientation", "정사각형 1:1 (1024x1024)")
    quality = st.session_state.get("image_quality", "low")

    if orientation.startswith("정사각형"):
        size = "1024x1024"
    elif orientation.startswith("가로형"):
        size = "1536x1024"
    else:
        size = "1024x1536"

    return size, quality


def generate_image(prompt: str):
    """주어진 프롬프트 그대로 이미지 생성"""
    if not prompt:
        return None

    size, quality = get_image_params()

    image_model_label = st.session_state.get("image_model_label", "OpenAI gpt-image-1")
    model = IMAGE_MODELS.get(image_model_label, "gpt-image-1")

    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    return resp.data[0].b64_json


def b64_to_bytes(b64_str: str):
    return base64.b64decode(b64_str)


def create_video_from_image_b64(
    image_b64: str,
    seconds_per_scene: float,
    fps: int = 30,
) -> tuple[bytes | None, str | None]:
    """
    단일 이미지(b64)로부터 영상 생성
    성공 시 (video_bytes, None)
    실패 시 (None, 에러메시지)
    """
    if imageio is None:
        return None, "IMAGEIO_MISSING"

    if not image_b64:
        return None, "NO_IMAGE"

    try:
        img_bytes = b64_to_bytes(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return None, f"IMAGE_DECODE_ERROR: {e}"

    frame = np.asarray(img)
    frames_per_scene = max(1, int(seconds_per_scene * fps))
    output_path = "imageking_output.mp4"

    try:
        writer = imageio.get_writer(output_path, fps=fps)  # imageio-ffmpeg 필요
    except Exception as e:
        return None, f"WRITER_ERROR: {e}"

    try:
        for _ in range(frames_per_scene):
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        return None, f"WRITE_FRAME_ERROR: {e}"

    try:
        with open(output_path, "rb") as f:
            return f.read(), None
    except Exception as e:
        return None, f"FILE_READ_ERROR: {e}"

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.markdown("### 🎬 IASA")
    st.markdown("---")

    st.markdown("#### 🖼 이미지 생성 모델")
    st.session_state["image_model_label"] = st.selectbox(
        "이미지 생성 모델",
        list(IMAGE_MODELS.keys()),
        index=list(IMAGE_MODELS.keys()).index(
            st.session_state.get("image_model_label", "OpenAI gpt-image-1")
        ),
    )

    # === 이미지 옵션: disclosure 그룹 ===
    with st.expander("🖼 이미지 옵션", expanded=True):
        st.session_state["image_orientation"] = st.radio(
            "비율 선택",
            ["정사각형 1:1 (1024x1024)", "가로형 3:2 (1536x1024)", "세로형 2:3 (1024x1536)"],
            index=["정사각형 1:1 (1024x1024)", "가로형 3:2 (1536x1024)", "세로형 2:3 (1024x1536)"].index(
                st.session_state.get("image_orientation", "정사각형 1:1 (1024x1024)")
            ),
        )

        st.session_state["image_quality"] = st.radio(
            "품질",
            ["low", "high"],
            index=["low", "high"].index(st.session_state.get("image_quality", "low")),
            horizontal=True,
        )

    # === 영상 생성 옵션: disclosure 그룹 ===
    with st.expander("🎥 영상 생성 옵션", expanded=True):
        st.session_state["video_model_label"] = st.selectbox(
            "영상 생성 모델",
            list(VIDEO_MODELS.keys()),
            index=list(VIDEO_MODELS.keys()).index(
                st.session_state.get("video_model_label", "이미지 1장 → MP4 (로컬 합성)")
            ),
        )

        st.session_state["seconds_per_scene"] = st.slider(
            "영상 길이 (초)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("seconds_per_scene", 3.0)),
            step=0.5,
        )

# =========================
# 메인 UI
# =========================
st.markdown(
    """
    <div>
        <div class="logo-badge">
            <span class="emoji">🎬</span>
            <span>IASA</span>
        </div>
        <div class="main-title">imageking</div>
        <div class="main-subtitle">
            하나의 이미지 프롬프트를 계속 변형해 보면서,<br>
            원하는 스타일을 찾는 실험용 이미지·영상 생성기입니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 중심 Disclosure (Expander)
# =========================
with st.expander("🧪 이미지 / 영상 생성", expanded=True):
    prompt_text = st.text_area(
        "이미지 프롬프트 (영어 권장)",
        height=220,
        value=st.session_state.get("prompt_text", ""),
        placeholder=(
            "예시:\n"
            "A Korean woman in her 20s with short hair,\n"
            "standing in a neon-lit street at night.\n"
            "50mm lens, medium shot, eye-level angle, cinematic framing.\n"
            "Cinematic realism, soft skin texture, subtle freckles.\n"
            "Rim lighting with pink and blue neon reflections.\n"
            "Moody and emotional atmosphere.\n"
            "Ultra-detailed, sharp focus, 8K resolution."
        ),
    )
    st.session_state["prompt_text"] = prompt_text

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        clicked_image = st.button("🖼 이미지 생성", type="primary", use_container_width=True)
    with col_btn2:
        clicked_video = st.button("🎬 영상 생성", type="secondary", use_container_width=True)

    # ---- 버튼 동작 처리 ----
    if clicked_image:
        if not prompt_text.strip():
            st.warning("이미지 프롬프트를 먼저 입력해주세요.")
        else:
            with st.spinner("이미지를 생성하는 중입니다..."):
                new_b64 = generate_image(prompt_text.strip())
            if new_b64:
                st.session_state["image_b64"] = new_b64
                st.session_state["video_bytes"] = None
                st.session_state["video_error_msg"] = None
                st.success("✅ 이미지가 생성되었습니다.")
            else:
                st.error("이미지 생성에 실패했습니다.")

    if clicked_video:
        if not st.session_state.get("image_b64"):
            st.warning("먼저 이미지를 생성한 후에 영상을 만들 수 있습니다.")
        else:
            if imageio is None:
                st.session_state["video_error_msg"] = (
                    "`imageio` 모듈이 없습니다. requirements.txt 에 `imageio` 와 `imageio-ffmpeg` 를 추가한 뒤 다시 배포해주세요."
                )
                st.session_state["video_bytes"] = None
            else:
                video_model_label = st.session_state.get("video_model_label", "이미지 1장 → MP4 (로컬 합성)")
                video_model = VIDEO_MODELS.get(video_model_label, "local_single_image_mp4")

                if video_model == "local_single_image_mp4":
                    seconds_per_scene = float(st.session_state.get("seconds_per_scene", 3.0))
                    with st.spinner("영상을 생성하는 중입니다..."):
                        video_bytes, err_msg = create_video_from_image_b64(
                            st.session_state.get("image_b64"),
                            seconds_per_scene=seconds_per_scene,
                            fps=30,
                        )
                    if video_bytes:
                        st.session_state["video_bytes"] = video_bytes
                        st.session_state["video_error_msg"] = None
                        st.success("🎬 영상이 생성되었습니다.")
                    else:
                        st.session_state["video_bytes"] = None
                        st.session_state["video_error_msg"] = (
                            "영상 생성 중 오류가 발생했습니다.\n\n"
                            "대부분은 `imageio-ffmpeg` 가 설치되지 않았거나 ffmpeg 플러그인을 찾지 못해서 생기는 문제입니다.\n"
                            "requirements.txt 에 `imageio-ffmpeg` 를 추가하고 다시 배포해 주세요.\n\n"
                            f"내부 오류 메시지: {err_msg}"
                        )
                else:
                    st.session_state["video_error_msg"] = "아직 구현되지 않은 영상 생성 모델입니다."
                    st.session_state["video_bytes"] = None

    # ---- 이미지 / 영상 결과 표시 (expander 안에서만) ----
    if st.session_state.get("image_b64"):
        st.markdown("---")
        st.markdown("#### 🖼 생성된 이미지")

        img_bytes = b64_to_bytes(st.session_state["image_b64"])
        # 이전 테이블에서 보이던 것처럼 column 폭에 맞게
        st.image(img_bytes, use_column_width=True)

        # 재생성 버튼 (같은 prompt_text로 다시 생성)
        if st.button("🔁 이 프롬프트로 다시 이미지 생성"):
            if not st.session_state.get("prompt_text", "").strip():
                st.warning("프롬프트가 비어 있습니다.")
            else:
                with st.spinner("이미지를 다시 생성하는 중입니다..."):
                    new_b64 = generate_image(st.session_state["prompt_text"].strip())
                if new_b64:
                    st.session_state["image_b64"] = new_b64
                    st.session_state["video_bytes"] = None
                    st.session_state["video_error_msg"] = None
                    st.success("✅ 이미지가 재생성되었습니다.")
                else:
                    st.error("이미지 재생성에 실패했습니다.")
            st.rerun()

    if st.session_state.get("video_bytes"):
        st.markdown("---")
        st.markdown("#### 🎬 생성된 영상 미리보기")
        st.video(st.session_state["video_bytes"])

        st.download_button(
            label="📥 영상 다운로드 (MP4)",
            data=st.session_state["video_bytes"],
            file_name="imageking_output.mp4",
            mime="video/mp4",
        )
    elif st.session_state.get("video_error_msg"):
        st.markdown("---")
        st.markdown("#### ⚠️ 영상 생성 오류")
        st.error(st.session_state["video_error_msg"])
