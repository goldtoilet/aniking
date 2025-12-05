import os
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# .env 로 환경변수 로드
# =========================
# 로컬 개발 시 .env 를 읽어옴 (Streamlit Cloud에서는 무시되고,
# 클라우드 환경변수만 사용됨)
load_dotenv()

# =========================
# 페이지 기본 설정 & 스타일
# =========================
st.set_page_config(
    page_title="AI 애니메이션 메이커",
    page_icon="🎬",
    layout="wide",
)

# 기본 textarea / 타이틀 스타일
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
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 환경변수에서 값 가져오기
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
# 세션 상태 기본값
# =========================
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("login_id", "")
st.session_state.setdefault("scenes", [])
st.session_state.setdefault("raw_script", "")

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
        user_id = st.text_input("아이디", value=st.session_state.get("login_id", ""), key="login_input_id")
        pw = st.text_input("비밀번호", type="password", key="login_input_pw")

        if st.button("로그인", type="primary", use_container_width=True):
            if not LOGIN_ID_ENV or not LOGIN_PW_ENV:
                st.error("서버에 로그인 정보가 설정되어 있지 않습니다. 관리자에게 문의하세요.")
            elif user_id == LOGIN_ID_ENV and pw == LOGIN_PW_ENV:
                st.session_state["logged_in"] = True
                st.session_state["login_id"] = user_id
                st.success("✅ 로그인 성공! 잠시 후 메인 화면으로 이동합니다.")
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


def generate_image(prompt: str, size: str = "512x512"):
    """OpenAI 이미지 하나 생성하고 base64 문자열 반환"""
    if not prompt:
        return None

    resp = client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json",   # ⭐ 필수!!!
    )
    return resp.data[0].b64_json

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


# =========================
# 사이드바
# =========================
with st.sidebar:
    st.markdown("### 🎬 AI 애니메이션 메이커")
    st.write(f"👤 로그인: **{st.session_state.get('login_id', '')}**")
    st.markdown("---")
    st.markdown("#### ⚙️ 향후 옵션")
    st.caption("- 스타일 프리셋 선택\n- 해상도 / 품질 옵션\n- 캐릭터 고정 설정 등")
    st.markdown("---")
    if st.button("로그아웃"):
        st.session_state["logged_in"] = False
        st.session_state["scenes"] = []
        st.session_state["raw_script"] = ""
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
            대본을 입력하고 원하는 스타일을 적용해, 문장별 프롬프트 → 이미지 → 영상으로 이어지는 파이프라인을 만드세요.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

raw_text = st.text_area(
    "여기에 대본을 붙여넣으세요.",
    height=260,
    value=st.session_state.get("raw_script", ""),
    placeholder="1\n문장… Shot on ...\n\n2\n문장… Shot on ...",
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    clicked_generate = st.button("이미지 생성", type="primary", use_container_width=True)
with col_btn2:
    st.button("영상 생성 (준비 중)", disabled=True, use_container_width=True)

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

# =========================
# 결과 테이블 출력
# =========================
scenes = st.session_state.get("scenes", [])

if scenes:
    st.subheader("문장별 프롬프트 및 이미지")

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

        # 한국어 문장
        cols[1].write(scene["korean"])

        # 영어 프롬프트
        cols[2].write(scene["prompt_en"])

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
else:
    st.info("대본을 입력하고 **이미지 생성** 버튼을 눌러주세요.")
