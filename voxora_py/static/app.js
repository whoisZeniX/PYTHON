(() => {
  const $ = (id) => document.getElementById(id);
  const MIME_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];

  let recorder = null;
  let stream = null;
  let audioContext = null;
  let analyser = null;
  let frame = null;
  let chunks = [];


  function showMic(state) {
    for (const name of ["idle", "recording", "processing"]) {
      const panel = $("mic-" + name);
      if (panel) panel.hidden = name !== state;
    }
  }

  function clearResults() {
    const area = $("results-area");
    if (area) area.replaceChildren();
  }

  function showError(message) {
    const node = $("error-template").content.firstElementChild.cloneNode(true);
    node.querySelector("[data-error-message]").textContent = message;
    $("results-area").replaceChildren(node);
  }

  function drawBars() {
    const canvas = $("mic-canvas");
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext("2d");
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const barWidth = (canvas.width / data.length) * 2.5;
    let x = 0;
    for (const value of data) {
      const barHeight = (value / 255) * canvas.height;
      ctx.fillStyle = "rgb(239, 68, 68)";
      ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
    frame = requestAnimationFrame(drawBars);
  }

  function stopVisualiser() {
    if (frame) cancelAnimationFrame(frame);
    frame = null;
    if (audioContext && audioContext.state !== "closed") audioContext.close();
    audioContext = null;
    analyser = null;
  }

  async function startRecording() {
    clearResults();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      audioContext.createMediaStreamSource(stream).connect(analyser);

      const mimeType = MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunks = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      };
      recorder.onstop = onRecordingStopped;
      recorder.start();

      showMic("recording");
      setTimeout(drawBars, 100);
    } catch (error) {
      console.error(error);
      stopVisualiser();
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null;
      showError("Microphone permission denied or unavailable.");
    }
  }

  function stopRecording() {
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      showMic("idle");
    }
  }

  async function onRecordingStopped() {
    const mimeType = recorder.mimeType || "audio/webm";
    const extension = mimeType.includes("ogg") ? "ogg" : "webm";
    const blob = new Blob(chunks, { type: mimeType });

    stopVisualiser();
    stream.getTracks().forEach((track) => track.stop());
    stream = null;

    if (blob.size === 0) {
      showError("No audio was recorded. Please try again.");
      return;
    }
    await upload(blob, extension);
  }


  async function upload(blob, extension) {
    showMic("processing");
    const form = new FormData();
    form.append("audio", blob, "recording." + extension);
    try {
      const response = await fetch("/ui/voice", { method: "POST", body: form });
      htmx.swap("#results-area", await response.text(), { swapStyle: "innerHTML" });
    } catch (error) {
      console.error(error);
      showError("Backend offline or request failed.");
    } finally {
      showMic("idle");
    }
  }


  function setDrawer(open) {
    const drawer = $("history-drawer");
    drawer.classList.toggle("translate-x-0", open);
    drawer.classList.toggle("translate-x-full", !open);
    $("history-close").hidden = !open;
    const fab = $("history-fab");
    if (fab) fab.hidden = open;
  }

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (target.closest("#mic-idle")) startRecording();
    else if (target.closest("#mic-recording")) stopRecording();
    else if (target.closest("#history-fab")) setDrawer(true);
    else if (target.closest("#history-close")) setDrawer(false);
  });
})();
