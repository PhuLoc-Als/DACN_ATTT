// Hàm vẽ kết quả ra vòng tròn + box
function renderResult(r) {
    const loading = document.getElementById("loading");
    const content = document.getElementById("content");
    loading.style.display = "none";
    content.style.display = "block";

    // ===== 1) Vẽ vòng tròn % =====
    const scoreRaw = Number.parseFloat(r.score);
    let score = Number.isFinite(scoreRaw) ? scoreRaw : 0;
    if (score < 0) score = 0;
    if (score > 1) score = 1;

    const percent = Math.round(score * 100);
    const scoreText = document.getElementById("score-text");
    const progressCircle = document.querySelector(".progress");

    const radius = 54;
    const circumference = 2 * Math.PI * radius;

    progressCircle.style.strokeDasharray = `${circumference}`;
    const offset = circumference - (percent / 100) * circumference;
    progressCircle.style.strokeDashoffset = offset;

    // Chọn màu
    let color = "green";
    const label = r.label || "unknown";

    if (label === "phish") color = "red";
    else if (score >= 0.45) color = "orange";

    progressCircle.style.stroke = color;
    scoreText.textContent = `${percent}%`;

    // ===== 2) Box thông tin =====
    const box = document.getElementById("result-box");

    const safeUrl = r.url || "(không rõ)";
    const safeLabel = (label === "phish" || label === "legit" || label === "error")
        ? label
        : "unknown";

    box.innerHTML = `
        <b>URL:</b>
        <div class="url-text">${safeUrl}</div>
        <br>
        <b>Tình trạng:</b> <span style="color:${color}">${safeLabel}</span><br>
        <b>Score raw:</b> ${Number.isFinite(scoreRaw) ? scoreRaw.toFixed(3) : "N/A"}
    `;

    // ===== 3) Link xem chi tiết =====
    const detailBtn = document.getElementById("detail-btn");
    detailBtn.href = "http://127.0.0.1:5000/detail?url=" + encodeURIComponent(safeUrl);
}

document.addEventListener("DOMContentLoaded", () => {
    const loading = document.getElementById("loading");
    const content = document.getElementById("content");

    // Reset UI mỗi lần mở popup
    loading.style.display = "block";
    loading.textContent = "Đang kiểm tra...";
    content.style.display = "none";

    chrome.storage.local.get(["lastResult", "checking"], (data) => {
        const checking = !!data.checking;
        const r = data.lastResult || null;

        // Nếu đang kiểm tra URL (server đang xử lý)
        if (checking && !r) {
            loading.textContent = "Đang kiểm tra...";
            loading.style.display = "block";
            content.style.display = "none";
            return;
        }

        // Nếu có kết quả rồi → hiển thị luôn,
        // giữ nguyên cho tới khi có request mới
        if (r) {
            renderResult(r);
            return;
        }

        // Không có gì để hiển thị (tab mới / chrome:// )
        loading.textContent = "Không có dữ liệu để hiển thị.";
        loading.style.display = "block";
        content.style.display = "none";
    });
});
