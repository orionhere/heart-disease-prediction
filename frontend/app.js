document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const submitBtn = document.getElementById("submit-btn");
    const resultContainer = document.getElementById("result-container");
    const riskGauge = document.getElementById("risk-gauge");
    const riskPercentage = document.getElementById("risk-percentage");
    const resultTitle = document.getElementById("result-title");
    const resultDesc = document.getElementById("result-desc");
    const resetBtn = document.getElementById("reset-btn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Enter loading state
        submitBtn.classList.add("loading");
        submitBtn.disabled = true;

        // Gather form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Ensure numeric types
        const payload = {};
        for (const [key, value] of Object.entries(data)) {
            payload[key] = Number(value);
        }

        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Prediction failed");
            }

            const result = await response.json();
            showResult(result);

        } catch (error) {
            console.error(error);
            alert("Error: " + error.message + "\nIs the FastAPI backend running?");
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    });

    resetBtn.addEventListener("click", () => {
        form.reset();
        resultContainer.classList.add("hidden");
        form.classList.remove("hidden");
        submitBtn.classList.remove("loading");
        submitBtn.disabled = false;
        riskGauge.style.background = `conic-gradient(var(--text-muted) 0%, transparent 0%)`;
    });

    function showResult(result) {
        form.classList.add("hidden");
        resultContainer.classList.remove("hidden");

        const isHighRisk = result.prediction === 1;
        const prob = result.probability !== null ? result.probability : (isHighRisk ? 0.85 : 0.15);
        const probPct = Math.round(prob * 100);

        resultTitle.textContent = isHighRisk ? "Action Required" : "Assessment Favorable";
        resultTitle.className = isHighRisk ? "high-risk" : "low-risk";

        riskPercentage.textContent = `${probPct}%`;
        riskPercentage.className = isHighRisk ? "high-risk" : "low-risk";

        resultDesc.innerHTML = isHighRisk
            ? `Our model indicates a <strong>high likelihood</strong> of heart disease. We strongly recommend consulting a healthcare professional for a comprehensive evaluation.`
            : `Our model indicates a <strong>low probability</strong> of heart disease based on the provided metrics. Continue maintaining a healthy lifestyle.`;

        // Animate gauge
        setTimeout(() => {
            const color = isHighRisk ? "var(--danger)" : "var(--success)";
            riskGauge.style.background = `conic-gradient(${color} ${probPct}%, rgba(255,255,255,0.05) ${probPct}%)`;
            riskGauge.style.boxShadow = `0 0 20px ${color}40`;
        }, 100);
    }
});
