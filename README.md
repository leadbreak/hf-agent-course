# 🤖 Hugging Face Agent Course (Interactive Series)

대형 언어 모델(LLM)이 자율적인 에이전트로 진화하는 과정과 아키텍처를 학습하는 인터랙티브 기술 문서입니다. 순수 Vanilla JS와 CSS 애니메이션을 통해 프레임워크의 동작 원리를 직관적으로 시각화했습니다.

## 📂 Repository Structure

유지보수와 확장을 위해 에셋과 뷰(View)를 분리한 구조를 채택했습니다.

- `index.html`: 전체 커리큘럼을 확인할 수 있는 메인 랜딩 페이지입니다.
- `lectures/`: 각 강의별 HTML 문서가 위치합니다. (예: `lecture-01.html`)
- `assets/css/`: 전체 강의에 공통으로 적용되는 디자인 시스템 및 애니메이션 룰셋(`style.css`)이 포함되어 있습니다.
- `assets/js/`: 스크롤 인터랙션 및 ReAct 사이클 시뮬레이터 구동을 위한 스크립트(`main.js`)가 포함되어 있습니다.

## 🚀 How to Navigate & Run

**GitHub Pages로 보기 (권장)**
> [👉 강의 메인 페이지로 이동하기](https://leadbreak.github.io/hf-agent-course/)

**로컬 환경에서 실행하기**
별도의 빌드 과정(Node.js 등)이 필요 없는 순수 정적 웹사이트입니다.
1. 저장소를 클론(`git clone`)합니다.
2. `index.html`을 웹 브라우저로 실행하여 강의 목록을 확인하고 각 `lectures/` 페이지로 이동할 수 있습니다.