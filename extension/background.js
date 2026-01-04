// Khi người dùng chuyển tab hoặc đổi URL → gọi API Flask để kiểm tra
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url) {
        checkPhishing(tabId, tab.url);
    }
});

// ===== BỎ QUA CÁC URL NỘI BỘ CỦA TRÌNH DUYỆT =====
function isInternalURL(url) {
    return (
        url.startsWith("chrome://") ||
        url.startsWith("edge://") ||
        url.startsWith("brave://") ||
        url.startsWith("opera://") ||
        url.includes("127.0.0.1:5000")
    );
}

// ===== SET ICON CHUẨN =====
function setIcon(tabId, level) {
    const icons = {
        red: {
            16: "icons/icon_red.png",
            32: "icons/icon_red.png",
            48: "icons/icon_red.png",
            128: "icons/icon_red.png",
        },
        yellow: {
            16: "icons/icon_yellow.png",
            32: "icons/icon_yellow.png",
            48: "icons/icon_yellow.png",
            128: "icons/icon_yellow.png",
        },
        green: {
            16: "icons/icon_green.png",
            32: "icons/icon_green.png",
            48: "icons/icon_green.png",
            128: "icons/icon_green.png",
        },
    };

    chrome.action.setIcon({
        tabId: tabId,
        path: icons[level],
    });
}

// ===== HÀM CHÍNH =====
async function checkPhishing(tabId, url) {
    // Không check URL nội bộ
    if (isInternalURL(url)) {
        console.log("Skip internal page:", url);
        return;
    }

    // Báo cho popup biết: đang kiểm tra URL mới
    chrome.storage.local.set({
        checking: true,
        lastUrl: url
    });

    try {
        const response = await fetch("http://127.0.0.1:5000/api/check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url }),
        });

        const result = await response.json();

        // Đảm bảo luôn có score là số
        if (typeof result.score !== "number" || !Number.isFinite(result.score)) {
            result.score = 0.0;
        }

        // Đổi icon theo kết quả
        if (result.label === "phish") {
            setIcon(tabId, "red");
        } else if (result.score >= 0.45) {
            setIcon(tabId, "yellow");
        } else {
            setIcon(tabId, "green");
        }

        // Lưu cache để popup đọc
        chrome.storage.local.set({
            lastResult: result,
            checking: false
        });

    } catch (err) {
        console.error("API error:", err);

        // Lưu lại trạng thái lỗi để popup hiển thị
        chrome.storage.local.set({
            lastResult: {
                url: url,
                label: "error",
                score: 0,
                error: String(err)
            },
            checking: false
        });
    }
}
