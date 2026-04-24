document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Scroll Reveal Animation (공통)
    const revealElements = document.querySelectorAll('.reveal');
    const chartFills = document.querySelectorAll('.bar-fill');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                
                // 차트 섹션 진입 시 막대그래프 애니메이션 트리거
                if (entry.target.querySelector('.bar-fill')) {
                    const fills = entry.target.querySelectorAll('.bar-fill');
                    fills.forEach(bar => {
                        bar.style.width = bar.getAttribute('data-width');
                    });
                }
                // 한 번 실행 후 관찰 해제 (원할 경우 유지 가능)
                // observer.unobserve(entry.target); 
            }
        });
    }, { threshold: 0.15 });

    revealElements.forEach(el => observer.observe(el));

    // 2. ReAct Loop 시뮬레이터 (lecture-01.html 전용)
    const termBody = document.getElementById('term-content');
    if (termBody) {
        const nodes = {
            thought: document.getElementById('node-thought'),
            action: document.getElementById('node-action'),
            obs: document.getElementById('node-obs')
        };
        
        const logs = [
            { id: 'thought', text: "[Thought] 데이터 전처리를 위해 Pandas 라이브러리 사용 구조를 기획합니다." },
            { id: 'action', text: "[Action] Python 인터프리터에 df.groupby() 연산 코드를 생성하여 실행합니다." },
            { id: 'obs', text: "[Observation] 실행 성공. 그룹화된 데이터 프레임의 헤더를 반환받았습니다." }
        ];

        let currentIdx = 0;
        let charIdx = 0;
        let isTyping = false;

        function typeWriter(text, callback) {
            isTyping = true;
            termBody.innerHTML = '';
            charIdx = 0;
            
            function type() {
                if (charIdx < text.length) {
                    termBody.innerHTML += text.charAt(charIdx);
                    charIdx++;
                    setTimeout(type, 30); // 타이핑 속도
                } else {
                    isTyping = false;
                    setTimeout(callback, 2000); // 출력 완료 후 대기 시간
                }
            }
            type();
        }

        function runLoop() {
            if (isTyping) return;

            // 이전 활성화 노드 리셋
            Object.values(nodes).forEach(n => n.classList.remove('active'));
            
            // 현재 노드 활성화 및 타이핑 시작
            const step = logs[currentIdx];
            nodes[step.id].classList.add('active');
            
            typeWriter(step.text, () => {
                currentIdx = (currentIdx + 1) % logs.length;
                runLoop(); // 다음 스텝 무한 루프
            });
        }

        // 초기 실행
        setTimeout(runLoop, 1000);
    }
});