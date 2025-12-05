import os
import io
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from PIL import Image
import imageio.v2 as imageio

# =========================
# .env 로 환경변수 로드 (로컬 개발용)
# =========================
load_dotenv()

# =========================
# 페이지 기본 설정 & 스타일
# =========================
st.set_page_config(
    page_title="AI 애니메이션 메이커",
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
LOGIN_ID_ENV = get_env("LOGIN_ID", "")
LOGIN_PW_ENV = get_env("LOGIN_PW", "")

if not GPT_API_KEY:
    st.error("❌ GPT_API_KEY 가 설정되어 있지 않습니다. .env 또는 환경변수를 확인해주세요.")
    st.stop()

client = OpenAI(api_key=GPT_API_KEY)

# =========================
# 이미지 / 영상 모델 프리셋
# =========================
IMAGE_MODELS = {
    "OpenAI gpt-image-1": "gpt-image-1",
    # 향후 다른 이미지 모델 추가 가능
}

VIDEO_MODELS = {
    "이미지 시퀀스 → MP4 (로컬 합성)": "local_sequence_mp4",
    # 향후 AI 비디오 모델 추가 가능
}

# =========================
# 스타일 프리셋 정의
# =========================
STYLE_PRESETS = {
    "다큐 + 사실적 배경": (
        "Style Wrapper:\n"
        "Background: documentary-style, semi-realistic environment, neutral color grading, "
        "soft cinematic lighting, subtle film grain.\n"
        "Characters: realistic human figures or crowds that match the story context.\n"
        "Camera: wide cinematic framing with natural depth and gentle atmospheric haze.\n"
    ),
    "다큐 + 스틱맨 설명 캐릭터": (
        "Style Wrapper:\n"
        "Background: semi-realistic cinematic environment, dramatic lighting, soft shadows, mild film grain, "
        "dystopian or documentary atmosphere.\n"
        "Characters: simple 2D stickman drawn with thick black outlines, white circular face, small black-dot eyes, "
        "line-based limbs, cartoon-like contrast against the realistic background.\n"
        "Camera: wide cinematic framing, slight depth of field, subtle atmospheric haze.\n"
    ),
    "풀 2D 애니메이션": (
        "Style Wrapper:\n"
        "Background: flat 2D animation style, pastel colors, simple geometric buildings and props.\n"
        "Characters: cute 2D stickman-style characters with thick black outlines and expressive poses.\n"
        "Camera: simple animation-style wide shot, clean composition, no film grain.\n"
    ),
}

# =========================
# 세션 상태 기본값
# =========================
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("login_id", "")
st.session_state.setdefault("scenes", [])
st.session_state.setdefault("raw_script", "")

st.session_state.setdefault("style_preset", "다큐 + 스틱맨 설명 캐릭터")
st.session_state.setdefault("lock_character", True)

st.session_state.setdefault("image_model_label", "OpenAI gpt-image-1")
st.session_state.setdefault("image_orientation", "정사각형 1:1 (1024x1024)")
st.session_state.setdefault("image_quality", "low")

st.session_state.setdefault("video_model_label", "이미지 시퀀스 → MP4 (로컬 합성)")
st.session_state.setdefault("seconds_per_scene", 3.0)
st.session_state.setdefault("video_bytes", None)

# =========================
# 로그인 화면
# =========================
def login_screen():
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;">
            <div class="logo-badge">
                <span class="emoji">🎬</span>
                <span>AI Animation Maker</span>
            </div>
            <div class="main-title">로그인이 필요합니다</div>
            <div class="main-subtitle">
                등록된 계정으로 로그인 후 AI 애니메이션 메이커를 사용할 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        user_id = st.text_input(
            "아이디",
            value=st.session_state.get("login_id", ""),
            key="login_input_id",
        )
        pw = st.text_input("비밀번호", type="password", key="login_input_pw")

        if st.button("로그인", type="primary", use_container_width=True):
            if not LOGIN_ID_ENV or not LOGIN_PW_ENV:
                st.error("서버에 로그인 정보(LOGIN_ID, LOGIN_PW)가 설정되어 있지 않습니다.")
            elif user_id == LOGIN_ID_ENV and pw == LOGIN_PW_ENV:
                st.session_state["logged_in"] = True
                st.session_state["login_id"] = user_id
                st.success("✅ 로그인 성공! 메인 화면으로 이동합니다.")
                st.rerun()
            else:
                st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")


# 로그인 체크
if not st.session_state.get("logged_in", False):
    login_screen()
    st.stop()

# =========================
# 유틸 함수들
# =========================
def parse_script(text: str):
    """
    대본 텍스트를 번호 / 한국어 문장 / 영어 프롬프트로 파싱.
    형식 예:
    1
    한국어문장… Shot on ...
    2
    한국어문장… Shot on ...
    """
    scenes = []

    # 번호로 시작하는 블록 단위로 분리
    pattern = r'(\d+)\s*\n(.+?)(?=\n\d+\s*\n|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for num, block in matches:
        block = block.strip()

        # 특수 줄바꿈( )도 일반 줄바꿈으로 치환
        block = block.replace("\u2028", "\n")

        if "Shot on" in block:
            ko_part, en_part = block.split("Shot on", 1)
            korean = ko_part.strip()
            english_prompt = "Shot on" + en_part.strip()
        else:
            korean = block.strip()
            english_prompt = ""

        scenes.append(
            {
                "id": int(num),
                "korean": korean,
                "prompt_en": english_prompt,
                "image_b64": None,
            }
        )
    return scenes


def get_image_params():
    """사이드바에서 선택한 옵션을 실제 size/quality 값으로 변환"""
    orientation = st.session_state.get("image_orientation", "정사각형 1:1 (1024x1024)")
    quality = st.session_state.get("image_quality", "low")

    if orientation.startswith("정사각형"):
        size = "1024x1024"
    elif orientation.startswith("가로형"):
        size = "1536x1024"  # 3:2 가로
    else:
        size = "1024x1536"  # 2:3 세로

    return size, quality


def build_full_prompt(base_prompt: str) -> str:
    """스타일 프리셋 + 캐릭터 고정 옵션을 포함한 최종 프롬프트 생성"""
    style_name = st.session_state.get("style_preset", "다큐 + 스틱맨 설명 캐릭터")
    style_wrapper = STYLE_PRESETS.get(style_name, "")

    lock_char = st.session_state.get("lock_character", False)
    if lock_char:
        # 스틱맨을 항상 등장시키는 추가 설명
        style_wrapper += (
            "\nThe main character is a recurring simple 2D stickman narrator with a white circular face "
            "and small black-dot eyes, always present somewhere in the scene, explaining or reacting to the situation.\n"
        )

    if style_wrapper:
        return style_wrapper + "\nScene:\n" + base_prompt
    else:
        return base_prompt


def generate_image(prompt: str):
    """OpenAI 이미지 하나 생성하고 base64 문자열 반환"""
    if not prompt:
        return None

    size, quality = get_image_params()
    full_prompt = build_full_prompt(prompt)

    image_model_label = st.session_state.get("image_model_label", "OpenAI gpt-image-1")
    model = IMAGE_MODELS.get(image_model_label, "gpt-image-1")

    resp = client.images.generate(
        model=model,
        prompt=full_prompt,
        size=size,
        quality=quality,  # low / high
        n=1,
    )
    b64_str = resp.data[0].b64_json  # base64 인코딩된 PNG
    return b64_str


def bulk_generate_images(scenes, max_workers: int = 4):
    """여러 장을 병렬로 생성"""
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


def create_video_from_scenes(scenes, seconds_per_scene: float, fps: int = 30) -> bytes | None:
    """
    이미지가 들어있는 scenes 리스트를 사용해 MP4 영상을 생성하고, 영상의 바이너리(bytes)를 반환.
    scene["image_b64"] 가 있는 항목만 사용.
    """
    images = []
    for scene in scenes:
        if not scene.get("image_b64"):
            continue
        img_bytes = b64_to_bytes(scene["image_b64"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(img)

    if not images:
        return None

    frames_per_scene = max(1, int(seconds_per_scene * fps))

    output_path = "aniking_output.mp4"
    writer = imageio.get_writer(output_path, fps=fps)
    for img in images:
        frame = imageio.asarray(img)
        for _ in range(frames_per_scene):
            writer.append_data(frame)
    writer.close()

    with open(output_path, "rb") as f:
        return f.read()


# =========================
# 사이드바 (스타일 / 옵션 / 모델 / 로그아웃)
# =========================
with st.sidebar:
    st.markdown("### 🎬 AI 애니메이션 메이커")
    st.write(f"👤 로그인: **{st.session_state.get('login_id', '')}**")
    st.markdown("---")

    # 이미지 생성 모델 선택
    st.markdown("#### 🖼 이미지 생성 모델")
    st.session_state["image_model_label"] = st.selectbox(
        "이미지 생성 모델",
        list(IMAGE_MODELS.keys()),
        index=list(IMAGE_MODELS.keys()).index(
            st.session_state.get("image_model_label", "OpenAI gpt-image-1")
        ),
    )

    st.markdown("#### 🎨 스타일 프리셋")
    st.session_state["style_preset"] = st.selectbox(
        "스타일 선택",
        list(STYLE_PRESETS.keys()),
        index=list(STYLE_PRESETS.keys()).index(
            st.session_state.get("style_preset", "다큐 + 스틱맨 설명 캐릭터")
        ),
    )

    st.markdown("#### 🧍 캐릭터 고정")
    st.session_state["lock_character"] = st.checkbox(
        "2D 스틱맨 설명 캐릭터 항상 포함",
        value=st.session_state.get("lock_character", True),
    )

    st.markdown("#### 🖼 이미지 옵션")
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

    st.markdown("#### 🎥 영상 생성 옵션")
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

    st.markdown("---")
    if st.button("로그아웃"):
        st.session_state["logged_in"] = False
        st.session_state["scenes"] = []
        st.session_state["raw_script"] = ""
        st.session_state["video_bytes"] = None
        st.rerun()

# =========================
# 메인 UI
# =========================
st.markdown(
    """
    <div>
        <div class="logo-badge">
            <span class="emoji">🎬</span>
            <span>AI Animation Maker</span>
        </div>
        <div class="main-title">AI 애니메이션 메이커</div>
        <div class="main-subtitle">
            대본을 입력하고, 문장별 프롬프트를 기반으로 이미지를 벌크로 생성한 뒤,
            장면들을 이어붙여 영상까지 자동으로 만들어보세요.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

raw_text = st.text_area(
    "여기에 대본을 붙여넣으세요.",
    height=260,
    value=st.session_state.get("raw_script", ""),
    placeholder="1\n한국어 문장… Shot on ...\n\n2\n한국어 문장… Shot on ...",
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    clicked_generate = st.button("이미지 생성", type="primary", use_container_width=True)
with col_btn2:
    clicked_video = st.button("영상 생성", type="secondary", use_container_width=True)

# =========================
# 이미지 생성 버튼 동작
# =========================
if clicked_generate:
    if not raw_text.strip():
        st.warning("대본을 먼저 입력해주세요.")
    else:
        scenes = parse_script(raw_text)
        if not scenes:
            st.error("대본 형식을 인식하지 못했습니다. 번호와 문장 형식을 다시 확인해주세요.")
        else:
            st.session_state["raw_script"] = raw_text
            st.session_state["scenes"] = scenes

            with st.spinner("이미지를 벌크로 생성 중입니다..."):
                bulk_generate_images(st.session_state["scenes"], max_workers=4)

            st.success("✅ 대본이 자동으로 분류되고 이미지가 생성되었습니다.")
            # 새로 이미지를 만들면 이전 영상은 초기화
            st.session_state["video_bytes"] = None

# 최신 scenes 반영
scenes = st.session_state.get("scenes", [])

# =========================
# 영상 생성 버튼 동작
# =========================
if clicked_video:
    if not scenes or not any(s.get("image_b64") for s in scenes):
        st.warning("먼저 이미지를 생성한 후에 영상을 만들 수 있습니다.")
    else:
        video_model_label = st.session_state.get("video_model_label", "이미지 시퀀스 → MP4 (로컬 합성)")
        video_model = VIDEO_MODELS.get(video_model_label, "local_sequence_mp4")

        if video_model == "local_sequence_mp4":
            seconds_per_scene = float(st.session_state.get("seconds_per_scene", 3.0))
            with st.spinner("영상을 생성하는 중입니다..."):
                video_bytes = create_video_from_scenes(scenes, seconds_per_scene=seconds_per_scene, fps=30)
            if video_bytes:
                st.session_state["video_bytes"] = video_bytes
                st.success("🎬 영상이 생성되었습니다.")
            else:
                st.error("영상 생성에 사용할 이미지가 없습니다.")
        else:
            st.error("아직 구현되지 않은 영상 생성 모델입니다.")

# =========================
# 결과 테이블 출력 (container + 스크롤 박스)
# =========================
if scenes:
    st.subheader("문장별 프롬프트 및 이미지")

    with st.container():
        st.markdown('<div class="results-container">', unsafe_allow_html=True)

        # 헤더
        header_cols = st.columns([0.5, 2, 2, 1, 0.9])
        header_cols[0].markdown("**번호**")
        header_cols[1].markdown("**원본문장**")
        header_cols[2].markdown("**생성된 영어 프롬프트**")
        header_cols[3].markdown("**이미지**")
        header_cols[4].markdown("**조작**")

        st.markdown("---")

        # 각 행
        for i, scene in enumerate(scenes):
            cols = st.columns([0.5, 2, 2, 1, 0.9])

            # 번호
            cols[0].write(scene["id"])

            # 한국어 문장 (작은 폰트)
            korean_html = scene["korean"].replace("\n", "<br>")
            cols[1].markdown(
                f'<div class="small-text-cell">{korean_html}</div>',
                unsafe_allow_html=True,
            )

            # 영어 프롬프트 (작은 폰트)
            prompt_html = scene["prompt_en"].replace("\n", "<br>")
            cols[2].markdown(
                f'<div class="small-text-cell">{prompt_html}</div>',
                unsafe_allow_html=True,
            )

            # 이미지
            if scene["image_b64"]:
                img_bytes = b64_to_bytes(scene["image_b64"])
                cols[3].image(img_bytes, use_column_width=True)
            else:
                cols[3].write("아직 이미지 없음")

            # 재생성 버튼
            if cols[4].button("재 생성", key=f"regen_{scene['id']}"):
                with st.spinner(f"{scene['id']}번 이미지를 다시 생성 중..."):
                    new_b64 = generate_image(scene["prompt_en"])
                    st.session_state["scenes"][i]["image_b64"] = new_b64
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("대본을 입력하고 **이미지 생성** 버튼을 눌러주세요.")

# =========================
# 생성된 영상 미리보기 / 다운로드
# =========================
if st.session_state.get("video_bytes"):
    st.subheader("🎬 생성된 영상 미리보기")
    st.video(st.session_state["video_bytes"])

    st.download_button(
        label="📥 영상 다운로드 (MP4)",
        data=st.session_state["video_bytes"],
        file_name="aniking_output.mp4",
        mime="video/mp4",
    )
