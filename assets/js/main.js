document.addEventListener('DOMContentLoaded', () => {
    // 1. Dynamic Scroll Reveal Configuration
    const revealElements = document.querySelectorAll('.reveal');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                // 해당 카드 내부에 데이터 그래프 바가 존재하면 폭 강제 설정
                const bars = entry.target.querySelectorAll('.bar-fill');
                bars.forEach(bar => {
                    bar.style.width = bar.getAttribute('data-width');
                });
            }
        });
    }, { threshold: 0.12 });
    revealElements.forEach(el => observer.observe(el));

    // 2. ReAct Loop Execution Simulator Logic
    const termContent = document.getElementById('term-content');
    if (termContent) {
        const nodes = {
            thought: document.getElementById('node-thought'),
            action: document.getElementById('node-action'),
            obs: document.getElementById('node-obs')
        };

        const steps = [
            { id: 'thought', text: ">> [Thought Engine] 목표: '런던 날씨 파악 및 가공'. 내부 추론 결과, 현재 지식 소스(Knowledge Cut-off) 외부의 실시간 지표이므로 연동된 외부 API 도구 레이어인 'get_weather' 함수를 호출하기로 결정함." },
            { id: 'action', text: ">> [Action Pipeline] 명세 포맷 검증 완료. 시스템 정지 토큰(Stop Sequence)을 발동하여 LLM 토큰 출력을 멈추고 파서(Parser)를 가동함. 호출 스트림: get_weather(location='London') 실행." },
            { id: 'obs', text: ">> [Observation Field] 외부 날씨 API 서버 응답 수신 성공. 환경 데이터 확보: {'weather': 'sunny', 'temp': 'low'}. 해당 리턴 로그를 대화형 프롬프트 문자열 최하단에 강제 강제 주입(Injection)함." }
        ];

        let loopIdx = 0;
        let charIdx = 0;
        let isWriting = false;

        function renderText(targetText, onComplete) {
            isWriting = true;
            termContent.innerHTML = '';
            charIdx = 0;

            function write() {
                if (charIdx < targetText.length) {
                    termContent.innerHTML += targetText.charAt(charIdx);
                    charIdx++;
                    setTimeout(write, 20);
                } else {
                    isWriting = false;
                    setTimeout(onComplete, 2500); // 웅장한 가독성을 위한 지연
                }
            }
            write();
        }

        function triggerNextStep() {
            if (isWriting) return;
            Object.values(nodes).forEach(n => n.classList.remove('active'));
            
            const currentStep = steps[loopIdx];
            nodes[currentStep.id].classList.add('active');
            
            renderText(currentStep.text, () => {
                loopIdx = (loopIdx + 1) % steps.length;
                triggerNextStep();
            });
        }
        setTimeout(triggerNextStep, 800);
    }
});