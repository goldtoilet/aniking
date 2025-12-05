import os
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI, BadRequestError, RateLimitError


# =========================
# 1. 기본 설정 & 클라이언트
# =========================

st.set_page_config(
    page_title="Aniking - 스크립트 투 이미지",
    page_icon="🎬",
    layout="wide",
)

API_KEY = os.getenv("GPT_API_KEY")
LOGIN_ID_ENV = os.getenv("LOGIN_ID")
LOGIN_PW_ENV = os.getenv("LOGIN_PW")

client = OpenAI(api_key=API_KEY)

# 세션 기본값
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("login_id", "")
st.session_state.setdefault("login_pw", "")
st.session_state.setdefault("scenes", [])  # [{"id":.., "text":.., "prompt":.., "image_b64":..}, ...]


# =========================
# 2. 유틸 함수
# =========================

def show_image_from_b64(b64_str: str):
    """base64 문자열을 실제 이미지로 렌더링"""
    if not b64_str:
        return
    try:
        img_bytes = base64.b64decode(b64_str)
        st.image(img_bytes)
    except Exception as e:
        st.error(f"이미지 디코딩 중 오류: {e}")


# =========================
# 3. OpenAI 이미지 생성
# =========================

def generate_image(prompt: str, size: str = "1024x1024"):
    """
    gpt-image-1 모델로 이미지 1장을 생성하고 base64 문자열 반환.
    prompt 가 비어 있으면 None 반환.
    """
    if not prompt or prompt.strip() == "":
        return None

    # gpt-image-1 에서 허용되는 사이즈
    valid_sizes = ("1024x1024", "1024x1536", "1536x1024", "auto")
    if size not in valid_sizes:
        size = "1024x1024"

    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size,
            # gpt-image-1 은 항상 b64_json 을 반환하므로
            # response_format 을 따로 줄 필요 없음 (주면 에러날 수 있음)
        )
        b64_str = resp.data[0].b64_json
        return b64_str

    except BadRequestError as e:
        st.error(f"❌ BadRequestError (요청 형식 오류): {e}")
        return None
    except RateLimitError as e:
        st.error(f"⏱️ RateLimitError (호출 한도 초과): {e}")
        return None
    except Exception as e:
        st.error(f"⚠️ 알 수 없는 이미지 생성 오류: {e}")
        return None


def bulk_generate_images(scenes, max_workers: int = 4, size: str = "1024x1024"):
    """
    scenes 리스트에 대해 병렬로 이미지를 생성.
    각 scene 은 {"id", "text", "prompt", "image_b64"} 구조를 기대.
    image_b64 필드에 base64 결과를 채워서 반환.
    """
    if not scenes:
        return scenes

    results = [None] * len(scenes)

    def _task(idx, prompt):
        b64 = generate_image(prompt, size=size)
        return idx, b64

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, scene in enumerate(scenes):
            prompt = scene.get("prompt") or scene.get("text") or ""
            if not prompt or prompt.strip() == "":
                results[idx] = None
                continue
            fut = executor.submit(_task, idx, prompt)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                idx_ret, b64 = fut.result()
                results[idx_ret] = b64
            except Exception as e:
                st.error(f"scene {idx} 이미지 생성 중 오류: {e}")
                results[idx] = None

    # 결과를 원래 scenes 에 병합
    for idx, b64 in enumerate(results):
        scenes[idx]["image_b64"] = b64

    return scenes


# =========================
# 4. 로그인 화면
# =========================

def login_screen():
    st.title("🔒 로그인 (Aniking)")

    col1, col2 = st.columns(2)
    with col1:
        login_id = st.text_input("아이디", value=st.session_state.get("login_id", ""))
    with col2:
        login_pw = st.text_input(
            "비밀번호",
            type="password",
            value=st.session_state.get("login_pw", ""),
        )

    if st.button("로그인"):
        if LOGIN_ID_ENV and LOGIN_PW_ENV:
            if login_id == LOGIN_ID_ENV and login_pw == LOGIN_PW_ENV:
                st.session_state["logged_in"] = True
                st.success("✅ 로그인 성공")
                st.experimental_rerun()
            else:
                st.error("❌ 아이디 또는 비밀번호가 올바르지 않습니다.")
        else:
            # 환경변수 미사용 시, 아무 값이나 넣으면 통과 (개발용)
            st.session_state["logged_in"] = True
            st.warning("환경변수가 없어 임시로 로그인을 통과시켰습니다.")
            st.experimental_rerun()

    st.session_state["login_id"] = login_id
    st.session_state["login_pw"] = login_pw


# =========================
# 5. 메인 앱 화면
# =========================

def app_main():
    st.title("🎬 Aniking - 스크립트 → 씬 → 이미지")

    # --- 좌측: 대본 입력 & 씬 생성 ---
    left, right = st.columns([1.1, 1.4])

    with left:
        st.subheader("1️⃣ 대본 입력")

        script_text = st.text_area(
            "한 줄당 한 씬으로 사용할 대본을 입력하세요.",
            height=200,
            placeholder="예)\n장면1 설명\n장면2 설명\n장면3 설명...",
        )

        if st.button("대본 → 씬 리스트 생성", type="primary"):
            lines = [ln.strip() for ln in script_text.splitlines() if ln.strip()]
            scenes = []
            for i, line in enumerate(lines, start=1):
                scenes.append(
                    {
                        "id": i,
                        "text": line,     # 원문
                        "prompt": line,   # 기본 프롬프트 (원하면 나중에 수정)
                        "image_b64": None,
                    }
                )
            st.session_state["scenes"] = scenes
            st.success(f"✅ 씬 {len(scenes)}개 생성 완료")

        st.markdown("---")

        st.subheader("2️⃣ 씬 프롬프트 편집")

        if not st.session_state["scenes"]:
            st.info("먼저 대본을 입력하고 씬 리스트를 생성하세요.")
        else:
            for scene in st.session_state["scenes"]:
                with st.expander(f"Scene {scene['id']}", expanded=False):
                    scene["text"] = st.text_input(
                        f"[{scene['id']}] 대본",
                        value=scene.get("text", ""),
                        key=f"text_{scene['id']}",
                    )
                    scene["prompt"] = st.text_area(
                        f"[{scene['id']}] 이미지 프롬프트 (영어/한국어 모두 가능)",
                        value=scene.get("prompt", ""),
                        key=f"prompt_{scene['id']}",
                        height=80,
                    )

    # --- 우측: 이미지 생성 & 미리보기 ---
    with right:
        st.subheader("3️⃣ 이미지 일괄 생성")

        if st.button("🖼 GPT-Image-1로 이미지 생성", type="primary"):
            if not st.session_state["scenes"]:
                st.warning("먼저 씬 리스트를 생성하세요.")
            else:
                with st.spinner("이미지 생성 중... (씬 수에 따라 시간이 걸릴 수 있습니다)"):
                    scenes_with_images = bulk_generate_images(
                        st.session_state["scenes"],
                        max_workers=4,
                        size="1024x1024",
                    )
                    st.session_state["scenes"] = scenes_with_images
                st.success("✅ 모든 씬에 대한 이미지 생성 완료")

        st.markdown("---")
        st.subheader("4️⃣ 결과 확인")

        if not st.session_state["scenes"]:
            st.info("아직 생성된 씬이 없습니다.")
        else:
            for scene in st.session_state["scenes"]:
                st.markdown(f"### Scene {scene['id']}")
                st.write(scene.get("text", ""))
                if scene.get("image_b64"):
                    show_image_from_b64(scene["image_b64"])
                else:
                    st.info("이미지가 아직 생성되지 않았습니다.")


# =========================
# 6. 진입점
# =========================

def main():
    # 로그인 안 되어 있으면 로그인 화면 먼저
    if not st.session_state.get("logged_in", False):
        login_screen()
    else:
        app_main()


if __name__ == "__main__":
    main()
