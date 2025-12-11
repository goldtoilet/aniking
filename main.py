import os
import io
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from PIL import Image
import numpy as np  # ← 추가

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
    /* 결과 테이블용 스크롤 박스 */
    .results-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 8px;
        border-radius: 8px;
        border: 1px solid #eee;
        background-color: #fafafa;
    }
    /* 테이블 안 텍스트 크기 줄이기 */
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
    "이미지 시퀀스 → MP4 (로컬 합성)": "local_sequence_mp4",
}

# =========================
# 세션 상태 기본값
# =========================
st.session_state.setdefault("scenes", [])
st.session_state.setdefault("base_prompt", "")
st.session_state.setdefault("prompt_variants_text", "")

st.session_state.setdefault("image_model_label", "OpenAI gpt-image-1")
st.session_state.setdefault("image_orientation", "정사각형 1:1 (1024x1024)")
st.session_state.setdefault("image_quality", "low")

st.session_state.setdefault("video_model_label", "이미지 시퀀스 → MP4 (로컬 합성)")
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


def bulk_generate_images(scenes, max_workers: int = 4):
    def _task(idx):
        prompt = scenes[idx]["prompt_en"]
        b64 = generate_image(prompt)
        return idx, b64

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_task, i) for i in range(len(scenes))]
        for fut in as_completed(futures):
            idx, b64 = fut.result()
            scenes[idx]["image_b64"] = b64


def b64_to_bytes(b64_str: str):
    return base64.b64decode(b64_str)


def create_video_from_scenes(
    scenes,
    seconds_per_scene: float,
    fps: int = 30,
) -> tuple[bytes | None, str | None]:
    """
    성공 시 (video_bytes, None)
    실패 시 (None, 에러메시지)
    """
    if imageio is None:
        return None, "IMAGEIO_MISSING"

    images = []
    for scene in scenes:
        if not scene.get("image_b64"):
            continue
        img_bytes = b64_to_bytes(scene["image_b64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    if not images:
        return None, "NO_IMAGES"

    frames_per_scene = max(1, int(seconds_per_scene * fps))
    output_path = "imageking_output.mp4"

    try:
        writer = imageio.get_writer(output_path, fps=fps)  # imageio-ffmpeg 필요
    except Exception as e:
        return None, f"WRITER_ERROR: {e}"

    try:
        for img in images:
            frame = np.asarray(img)
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


def build_scenes_from_prompt(base_prompt: str, variants_text: str):
    """
    기본 프롬프트 + 변형 리스트(줄바꿈)로 scenes 생성
    - 변형이 없으면 기본 프롬프트 1개만 사용
    - 변형이 여러 줄이면 각 줄마다 기본 프롬프트에 붙여서 하나의 장면으로 사용
    """
    scenes = []
    base_prompt = base_prompt.strip()

    if not base_prompt and not variants_text.strip():
        return scenes

    variant_lines = [ln.strip() for ln in variants_text.splitlines() if ln.strip()]

    # 변형이 없으면 기본 프롬프트만 1개
    if not variant_lines:
        combined_prompt = base_prompt
        scenes.append(
            {
                "id": 1,
                "korean": base_prompt,
                "prompt_en": combined_prompt,
                "image_b64": None,
            }
        )
        return scenes

    # 변형이 있으면 기본 프롬프트 + 각 변형 조합으로 여러 장면 생성
    for i, v in enumerate(variant_lines, start=1):
        if base_prompt:
            combined_prompt = f"{base_prompt}, {v}"
        else:
            combined_prompt = v

        korean_block = base_prompt + ("\n" + v if base_prompt else v)

        scenes.append(
            {
                "id": i,
                "korean": korean_block,
                "prompt_en": combined_prompt,
                "image_b64": None,
            }
        )

    return scenes

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
                st.session_state.get("video_model_label", "이미지 시퀀스 → MP4 (로컬 합성)")
            ),
        )

        st.session_state["seconds_per_scene"] = st.slider(
            "장면당 영상 길이 (초)",
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
            하나의 기본 이미지 프롬프트를 정해 두고,<br>
            여러 가지 변형 프롬프트를 실험하면서 원하는 이미지를 찾아보세요.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 기본 프롬프트 & 변형 리스트 입력 ---
base_prompt = st.text_input(
    "기본 이미지 프롬프트 (영어 권장)",
    value=st.session_state.get("base_prompt", ""),
    placeholder="예: A Korean woman in her 20s, standing in a neon-lit street at night, 50mm lens, cinematic framing",
)

variants_text = st.text_area(
    "프롬프트 변형 리스트 (줄바꿈으로 구분)",
    height=200,
    value=st.session_state.get("prompt_variants_text", ""),
    placeholder=(
        "각 줄마다 다른 변형 요소를 적어보세요.\n"
        "예)\n"
        "cinematic lighting, moody atmosphere\n"
        "sunset, orange and teal color grading\n"
        "top-down view, 35mm lens\n"
    ),
)

st.session_state["base_prompt"] = base_prompt
st.session_state["prompt_variants_text"] = variants_text

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    clicked_generate = st.button("이미지 생성", type="primary", use_container_width=True)
with col_btn2:
    clicked_video = st.button("영상 생성", type="secondary", use_container_width=True)

# =========================
# 이미지 생성 버튼 동작
# =========================
if clicked_generate:
    if not base_prompt.strip() and not variants_text.strip():
        st.warning("기본 프롬프트 또는 변형 프롬프트를 하나 이상 입력해주세요.")
    else:
        scenes = build_scenes_from_prompt(base_prompt, variants_text)
        if not scenes:
            st.error("프롬프트를 인식하지 못했습니다. 내용을 다시 확인해주세요.")
        else:
            st.session_state["scenes"] = scenes

            with st.spinner("이미지를 벌크로 생성 중입니다..."):
                bulk_generate_images(st.session_state["scenes"], max_workers=4)

            st.success("✅ 프롬프트가 장면으로 분리되고 이미지가 생성되었습니다.")
            st.session_state["video_bytes"] = None
            st.session_state["video_error_msg"] = None

scenes = st.session_state.get("scenes", [])

# =========================
# 영상 생성 버튼 동작
# =========================
if clicked_video:
    if not scenes or not any(s.get("image_b64") for s in scenes):
        st.warning("먼저 이미지를 생성한 후에 영상을 만들 수 있습니다.")
    else:
        if imageio is None:
            st.session_state["video_error_msg"] = (
                "`imageio` 모듈이 없습니다. requirements.txt 에 `imageio` 와 `imageio-ffmpeg` 를 추가한 뒤 다시 배포해주세요."
            )
            st.session_state["video_bytes"] = None
        else:
            video_model_label = st.session_state.get("video_model_label", "이미지 시퀀스 → MP4 (로컬 합성)")
            video_model = VIDEO_MODELS.get(video_model_label, "local_sequence_mp4")

            if video_model == "local_sequence_mp4":
                seconds_per_scene = float(st.session_state.get("seconds_per_scene", 3.0))
                with st.spinner("영상을 생성하는 중입니다..."):
                    video_bytes, err_msg = create_video_from_scenes(
                        scenes,
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

# ==========================
# 결과 테이블 (스크롤 컨테이너)
# =========================
if scenes:
    st.subheader("프롬프트 변형별 이미지 결과")

    with st.container():
        st.markdown('<div class="results-container">', unsafe_allow_html=True)

        header_cols = st.columns([0.5, 2, 2, 1, 0.9])
        header_cols[0].markdown("**번호**")
        header_cols[1].markdown("**기본 + 변형 프롬프트**")
        header_cols[2].markdown("**최종 전달 프롬프트**")
        header_cols[3].markdown("**이미지**")
        header_cols[4].markdown("**조작**")

        st.markdown("---")

        for i, scene in enumerate(scenes):
            cols = st.columns([0.5, 2, 2, 1, 0.9])

            cols[0].write(scene["id"])

            korean_html = scene["korean"].replace("\n", "<br>")
            cols[1].markdown(
                f'<div class="small-text-cell">{korean_html}</div>',
                unsafe_allow_html=True,
            )

            prompt_html = scene["prompt_en"].replace("\n", "<br>")
            cols[2].markdown(
                f'<div class="small-text-cell">{prompt_html}</div>',
                unsafe_allow_html=True,
            )

            if scene["image_b64"]:
                img_bytes = b64_to_bytes(scene["image_b64"])
                cols[3].image(img_bytes, use_column_width=True)
            else:
                cols[3].write("아직 이미지 없음")

            if cols[4].button("재 생성", key=f"regen_{scene['id']}"):
                with st.spinner(f"{scene['id']}번 이미지를 다시 생성 중..."):
                    new_b64 = generate_image(scene["prompt_en"])
                    st.session_state["scenes"][i]["image_b64"] = new_b64
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("기본 프롬프트와 변형 프롬프트를 입력하고 **이미지 생성** 버튼을 눌러주세요.")

# =========================
# 생성된 영상 / 오류 표시
# =========================
if st.session_state.get("video_bytes"):
    st.subheader("🎬 생성된 영상 미리보기")
    st.video(st.session_state["video_bytes"])

    st.download_button(
        label="📥 영상 다운로드 (MP4)",
        data=st.session_state["video_bytes"],
        file_name="imageking_output.mp4",
        mime="video/mp4",
    )
elif st.session_state.get("video_error_msg"):
    st.subheader("⚠️ 영상 생성 오류")
    st.error(st.session_state["video_error_msg"])
