document.addEventListener("DOMContentLoaded", () => {
    const progress = document.querySelector(".progress span");
    const revealItems = document.querySelectorAll(".reveal");
    const bars = document.querySelectorAll(".bar span");
    const radar = document.getElementById("agentRadar");
    const reactOutput = document.getElementById("reactOutput");
    const reactNodes = document.querySelectorAll(".cycle-node");

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function updateProgress() {
        if (!progress) return;
        const scrollable = document.documentElement.scrollHeight - window.innerHeight;
        const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
        progress.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
    }

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
        });
    }, { threshold: 0.14 });

    revealItems.forEach((item) => revealObserver.observe(item));

    const barObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            entry.target.style.width = entry.target.dataset.width || "0";
            barObserver.unobserve(entry.target);
        });
    }, { threshold: 0.7 });

    bars.forEach((bar) => barObserver.observe(bar));

    const reactSteps = {
        thought: {
            label: "Thought",
            text: "Thought: 사용자 목표에는 최신 날씨 정보가 필요하다. 내부 지식만으로 답하면 위험하므로 get_weather 도구를 호출한다."
        },
        action: {
            label: "Action",
            text: "Action: get_weather(city='Seoul')\nTool result: {'condition': 'rain likely', 'temp': '18-22C', 'risk': 'outdoor delay'}"
        },
        observation: {
            label: "Observation",
            text: "Observation: 비 예보가 있으므로 공지는 야외 진행과 실내 대안을 함께 안내해야 한다. 이제 최종 문장을 작성한다."
        }
    };

    let activeStep = "thought";
    let stepTimer;

    function typeText(text) {
        if (!reactOutput) return;
        if (prefersReducedMotion) {
            reactOutput.textContent = text;
            return;
        }

        reactOutput.textContent = "";
        let index = 0;

        function write() {
            reactOutput.textContent += text[index] || "";
            index += 1;
            if (index < text.length) window.setTimeout(write, 13);
        }

        write();
    }

    function setReactStep(step) {
        activeStep = step;
        reactNodes.forEach((node) => {
            node.classList.toggle("is-active", node.dataset.step === step);
        });
        typeText(reactSteps[step].text);
    }

    function startReactLoop() {
        if (!reactOutput || prefersReducedMotion) return;
        const order = ["thought", "action", "observation"];
        let idx = order.indexOf(activeStep);
        window.clearInterval(stepTimer);
        stepTimer = window.setInterval(() => {
            idx = (idx + 1) % order.length;
            setReactStep(order[idx]);
        }, 5200);
    }

    reactNodes.forEach((node) => {
        node.addEventListener("click", () => {
            setReactStep(node.dataset.step);
            startReactLoop();
        });
    });

    setReactStep(activeStep);
    startReactLoop();

    function drawRadarChart(progressRatio = 1) {
        if (!radar) return;
        const ctx = radar.getContext("2d");
        const width = radar.width;
        const height = radar.height;
        const centerX = width / 2;
        const centerY = height / 2 + 10;
        const radius = Math.min(width, height) * 0.34;
        const labels = ["자율성", "유연성", "도구 활용", "최신성", "예측성"];
        const agent = [92, 86, 94, 82, 52].map((value) => value * progressRatio);
        const program = [16, 28, 32, 35, 94].map((value) => value * progressRatio);

        ctx.clearRect(0, 0, width, height);
        ctx.font = "15px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (let ring = 1; ring <= 4; ring += 1) {
            const ringRadius = radius * (ring / 4);
            ctx.beginPath();
            labels.forEach((_, index) => {
                const angle = -Math.PI / 2 + (index * 2 * Math.PI) / labels.length;
                const x = centerX + Math.cos(angle) * ringRadius;
                const y = centerY + Math.sin(angle) * ringRadius;
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.strokeStyle = "rgba(255,255,255,0.12)";
            ctx.stroke();
        }

        labels.forEach((label, index) => {
            const angle = -Math.PI / 2 + (index * 2 * Math.PI) / labels.length;
            const x = centerX + Math.cos(angle) * (radius + 36);
            const y = centerY + Math.sin(angle) * (radius + 30);
            ctx.fillStyle = "#dce3ed";
            ctx.fillText(label, x, y);
        });

        function plot(values, color, fill) {
            ctx.beginPath();
            values.forEach((value, index) => {
                const angle = -Math.PI / 2 + (index * 2 * Math.PI) / values.length;
                const pointRadius = radius * (value / 100);
                const x = centerX + Math.cos(angle) * pointRadius;
                const y = centerY + Math.sin(angle) * pointRadius;
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.fillStyle = fill;
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.fill();
            ctx.stroke();
        }

        plot(program, "#66a8ff", "rgba(102,168,255,0.18)");
        plot(agent, "#ffd21e", "rgba(255,210,30,0.18)");

        ctx.textAlign = "left";
        ctx.fillStyle = "#ffd21e";
        ctx.fillText("Agent", 24, 28);
        ctx.fillStyle = "#66a8ff";
        ctx.fillText("Rule-based", 24, 52);
    }

    if (radar) {
        let hasDrawn = false;
        const radarObserver = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
                if (!entry.isIntersecting || hasDrawn) return;
                hasDrawn = true;

                if (prefersReducedMotion) {
                    drawRadarChart(1);
                    return;
                }

                const start = performance.now();
                function animate(now) {
                    const ratio = Math.min(1, (now - start) / 1000);
                    drawRadarChart(ratio);
                    if (ratio < 1) requestAnimationFrame(animate);
                }
                requestAnimationFrame(animate);
            });
        }, { threshold: 0.55 });

        radarObserver.observe(radar);
        drawRadarChart(0.05);
    }

    window.addEventListener("scroll", updateProgress, { passive: true });
    window.addEventListener("resize", () => drawRadarChart(1));
    updateProgress();
});
