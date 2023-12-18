const extractiveBtn = document.getElementById("summarise");
const abstractiveBtn = document.getElementById("abstractiveSummarise");

function performSummarization(extractive) {
    const btn = extractive ? extractiveBtn : abstractiveBtn;
    btn.disabled = true;
    btn.innerHTML = "Summarising...";

    // Get the active tab information
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
        const url = tabs[0].url;
        const endpoint = extractive ? "summary" : "abstractive_summary"; // Choose the appropriate endpoint
        const xhr = new XMLHttpRequest();
        xhr.open("GET", `http://127.0.0.1:5000/${endpoint}?url=${url}`, true);
        xhr.onload = function() {
            const text = xhr.responseText;
            const p = document.getElementById("output");
            p.innerHTML = text;
            btn.disabled = false;
            btn.innerHTML = `Summarise (${extractive ? 'Extractive' : 'Abstractive'})`;
        };
        xhr.send();
    });
}

extractiveBtn.addEventListener("click", function() {
    performSummarization(true);
});

abstractiveBtn.addEventListener("click", function() {
    performSummarization(false);
});
