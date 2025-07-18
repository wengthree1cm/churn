document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const modelInput = document.getElementById("model");
  const outputDiv = document.getElementById("output");

  outputDiv.innerHTML = "⏳ Uploading file and predicting...";

  if (!fileInput.files.length) {
    outputDiv.innerHTML = `<p style="color:red">Please upload a CSV file.</p>`;
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("model_type", modelInput.value);

  try {
    const response = await fetch("https://churn-prediction-udq1.onrender.com/predict", {
      method: "POST",
      body: formData,
      headers: {
        // 必须去掉 Content-Type，浏览器会自动设置 multipart/form-data boundary
        // "Content-Type": "multipart/form-data" ❌ 不能手动加这个
      },
    });

    const rawText = await response.text(); // 获取原始返回文本
    console.log("Raw Response:", rawText); // 👉 打印原始返回内容用于调试

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}: ${rawText}`);
    }

    let result;
    try {
      result = JSON.parse(rawText);  // 手动解析 JSON，避免格式问题抛错
    } catch (parseErr) {
      throw new Error(`Failed to parse JSON response: ${parseErr.message}`);
    }

    outputDiv.innerHTML = ""; // 清除 loading 信息

    if (result.error) {
      outputDiv.innerHTML = `<p style="color:red">⚠️ ${result.error}</p>`;
      return;
    }

    if (result.predictions && result.predictions.length > 0) {
      let table = "<table border='1'><thead><tr><th>Prediction</th></tr></thead><tbody>";
      result.predictions.forEach(pred => {
        table += `<tr><td>${pred}</td></tr>`;
      });
      table += "</tbody></table>";
      outputDiv.innerHTML = table;
    } else {
      outputDiv.innerHTML = "<p style='color:red'>❗No predictions returned.</p>";
    }

  } catch (error) {
    console.error("Fetch failed:", error);
    outputDiv.innerHTML = `<p style="color:red">❌ Fetch failed: ${error.message}</p>`;
  }
});
