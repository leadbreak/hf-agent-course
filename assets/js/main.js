document.addEventListener("DOMContentLoaded", () => {
    const progress = document.querySelector(".site-progress span");
    const revealItems = document.querySelectorAll(".reveal");
    const counters = document.querySelectorAll("[data-count]");

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
    }, { threshold: 0.16 });

    revealItems.forEach((item) => revealObserver.observe(item));

    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            const target = Number(entry.target.dataset.count || 0);
            const start = performance.now();

            function tick(now) {
                const progressRatio = Math.min(1, (now - start) / 850);
                entry.target.textContent = Math.round(target * progressRatio);
                if (progressRatio < 1) requestAnimationFrame(tick);
            }

            requestAnimationFrame(tick);
            counterObserver.unobserve(entry.target);
        });
    }, { threshold: 0.6 });

    counters.forEach((counter) => counterObserver.observe(counter));

    window.addEventListener("scroll", updateProgress, { passive: true });
    updateProgress();
});
