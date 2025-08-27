# 🎨 Generative AI Studio ✨

**광고 크리에이티브 제작을 위한 올인원 생성형 AI 플랫폼**

---

## 🚀 프로젝트 개요 (Project Overview)

이 프로젝트는 광고 에이전시를 위한 **웹 기반 생성형 AI 스튜디오**입니다. 사용자가 텍스트 프롬프트나 이미지를 입력하여 광고 제작에 필요한 다양한 크리에이티브 에셋(이미지, 비디오, 음성, 음악)을 생성, 편집, 분석할 수 있는 통합 플랫폼을 제공합니다.

Google Cloud의 최신 AI 모델(Gemini, Imagen, Veo 등)을 기반으로 하며, 아이디어 구상부터 광고 적합성 분석까지 광고 제작의 전 과정을 지원합니다.

---

## 🌟 주요 기능 (Key Features)

| 기능 | 설명 | 사용 모델 |
| :--- | :--- | :--- |
| 🍌 **나노바나나 에디터** | 여러 이미지와 프롬프트를 조합해 섬세한 편집/합성 | `gemini-2.5-flash-image` |
| 🖼️ **이미지 편집** | 프롬프트를 기반으로 이미지 수정 | `Imagen 3` |
| 🏞️ **이미지 생성** | 텍스트로 새로운 이미지 생성 | `Imagen 4` |
| 🔍 **이미지 업스케일링** | 이미지 해상도 4배 향상 | `Imagen 3` |
| 🎬 **비디오 생성** | 이미지와 프롬프트로 짧은 비디오 제작 | `Veo` |
| 🎵 **음악 생성** | 분위기/장르에 맞는 배경 음악 생성 | `Lyria` |
| 🗣️ **음성 생성** | 텍스트를 자연스러운 음성으로 변환 | `Google TTS` |
| 📊 **광고 적합성 분석** | 생성물의 광고 적합성 자동 분석 | `Vision API + Gemini` |
| ✨ **파인튜닝 시뮬레이션** | 특정 스타일을 학습한 것처럼 일관된 톤의 이미지 생성 | `Gemini + Imagen` |

---

## 📂 프로젝트 구조 (Project Structure)

```
/
├── backend/              # 벡엔드 (Python, Flask)
│   ├── main.py           # 핵심 API 로직
│   ├── requirements.txt  # Python 의존성 목록
│   └── .env              # 환경 변수 설정 파일
├── frontend/             # 프론트엔드 (HTML, CSS, JS)
│   ├── index.html        # 메인 UI 구조
│   ├── style.css         # UI 스타일
│   └── app.js            # 클라이언트 로직
├── Dockerfile            # 컨테이너 배포 설정
└── README.md             # 프로젝트 소개
```

-   **`backend/`**: Flask 기반의 백엔드 서버. Google Cloud AI 서비스와 연동하여 실제 생성 및 분석 작업을 처리하는 API를 제공합니다.
-   **`frontend/`**: 사용자가 상호작용하는 웹 인터페이스. 순수 HTML/CSS/JavaScript로 구성된 싱글 페이지 애플리케이션(SPA)입니다.
-   **`Dockerfile`**: 애플리케이션을 컨테이너화하여 Google Cloud Run과 같은 환경에 쉽게 배포할 수 있도록 정의한 파일입니다.

---

## 🛠️ 기술 스택 (Tech Stack)

-   **Backend**: `Python 3.12`, `Flask`, `Gunicorn`
-   **Frontend**: `HTML5`, `CSS3`, `JavaScript (ES6+)`
-   **Google Cloud Services**:
    -   Vertex AI (Gemini 2.5 Flash, Imagen 3/4, Veo, Lyria)
    -   AI Studio (Gemini 2.5 Flash Image)
    -   Cloud Storage
    -   Vision API
    -   Text-to-Speech API
-   **Deployment**: `Docker`, `Google Cloud Run`

---

## ⚙️ 실행 방법 (How to Run)

### 1. 로컬 환경 (Local Environment)

1.  **저장소 복제 및 디렉터리 이동**
    ```bash
    git clone https://github.com/ldu1225/gcp_generative_ai_studio.git
    cd gcp_generative_ai_studio
    ```

2.  **환경 변수 설정**
    - `backend/.env` 파일을 생성하고 아래 내용을 채웁니다.
    ```env
    PROJECT_ID="your-gcp-project-id"
    BUCKET_NAME="your-gcs-bucket-name"
    GOOGLE_API_KEY="your-ai-studio-api-key"
    ```

3.  **Python 가상환경 생성 및 활성화**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **의존성 설치**
    ```bash
    pip install -r backend/requirements.txt
    ```

5.  **백엔드 서버 실행**
    ```bash
    python backend/main.py
    ```

6.  **접속**: 웹 브라우저에서 `http://127.0.0.1:8080` 주소로 접속합니다.

### 2. Docker 환경 (Docker Environment)

1.  **Docker 이미지 빌드**
    ```bash
    docker build -t hsad-ai-studio .
    ```

2.  **Docker 컨테이너 실행**
    - (주의: Docker 실행 시 `.env` 파일의 환경 변수를 전달해야 합니다.)
    ```bash
    docker run -p 8080:8080 --env-file backend/.env hsad-ai-studio
    ```

3.  **접속**: 웹 브라우저에서 `http://localhost:8080` 주소로 접속합니다.

---

## 💡 개선 제안 (Suggestions for Improvement)

-   **프론트엔드 고도화**: React, Vue와 같은 모던 프레임워크를 도입하여 상태 관리 및 UI/UX 개선.
-   **비동기 처리 개선**: 비디오 생성과 같이 오래 걸리는 작업은 WebSocket 또는 Pub/Sub 기반으로 전환하여 사용자 경험 향상.
-   **인증 및 사용자 관리**: 사용자 계정 시스템을 도입하여 작업 내역 관리 및 개인화 기능 제공.
-   **테스트 코드 작성**: 안정적인 서비스 운영을 위한 단위 테스트 및 통합 테스트 코드 추가.
