import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// --- Global Timer for OnixExecutionTimer ---
const OnixGlobalTimer = {
    startTime: 0,
    intervalId: null,
    isRunning: false,
    activeNodes: new Set(),

    formatTime(ms) {
        if (ms < 0) ms = 0;
        const minutes = String(Math.floor(ms / 60000)).padStart(2, '0');
        const seconds = String(Math.floor((ms % 60000) / 1000)).padStart(2, '0');
        const milliseconds = String(ms % 1000).padStart(3, '0');
        return `${minutes}:${seconds}:${milliseconds}`;
    },
    
    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.startTime = Date.now();
        
        this.activeNodes.forEach(node => {
            if (node.timerDisplay) node.timerDisplay.style.color = '#aaffaa';
        });

        this.intervalId = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const timeString = this.formatTime(elapsed);
            this.activeNodes.forEach(node => {
                if (node.timerDisplay) node.timerDisplay.textContent = timeString;
            });
        }, 33);
    },

    stop() {
        if (!this.isRunning) return;
        this.isRunning = false;
        clearInterval(this.intervalId);
        
        const finalTime = Date.now() - this.startTime;
        const finalTimeString = this.formatTime(finalTime);
        
        this.activeNodes.forEach(node => {
            if (node.timerDisplay) {
                node.timerDisplay.textContent = finalTimeString;
                node.timerDisplay.style.color = 'var(--text-color, #7300ff)';
            }
            node.properties.elapsed_time_str = finalTimeString;
        });
    },

    registerNode(node) { this.activeNodes.add(node); },
    unregisterNode(node) { this.activeNodes.delete(node); },
};

// --- Helper functions for OnixProject ---
function hideInternalWidgets(node) {
  ["existing_project", "project_id"].forEach((name) => {
    const w = node.widgets?.find((w) => w.name === name);
    if (w) {
      w.hidden = true;
      w.computeSize = () => [0, -4];
    }
  });
}

function setNameDisabled(widget, disabled) {
  if (!widget) return;
  const el = widget.inputEl || widget.textareaEl || widget.element;
  widget.readOnly = !!disabled;
  if (el) {
    el.readOnly = !!disabled;
    el.disabled = !!disabled;
    el.tabIndex = disabled ? -1 : 0;
    const st = el.style;
    if (st) {
      st.pointerEvents = disabled ? "none" : "auto";
      st.opacity = disabled ? "0.7" : "1";
      st.cursor = disabled ? "not-allowed" : "";
    }
  }
  widget.disabled = !!disabled;
}

function sanitizeFileName(name) {
  return (name || "").replace(/[^a-zA-Z0-9._-]/g, "");
}

async function fetchProjectFiles() {
  const res = await fetch("/onix/projects", { credentials: "same-origin" });
  if (!res.ok) return ["none"];
  const data = await res.json();
  const files = Array.isArray(data.files) ? data.files.filter(Boolean) : [];
  return ["none", ...files];
}

function hookAfterExec(node, handler) {
  const orig = node.onExecuted?.bind(node);
  node.onExecuted = function (output) {
    try { handler(output); } catch (e) {}
    return orig ? orig(output) : undefined;
  };

  const s = app?.socket;
  if (s && !node.__vp_exec_hooked) {
    node.__vp_exec_hooked = true;
    s.on("executed", (msg) => {
      const nid = msg?.node_id ?? msg?.node;
      if (nid !== node.id) return;
      try { handler(msg?.output ?? msg); } catch (e) {}
    });
  }
}

// --- Main Extension Registration ---
app.registerExtension({
  name: "OnixProjectNode",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Handling OnixProject Node
    if (nodeData.name === "OnixProject") {
      const origOnNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

        const getW = (n) => this.widgets?.find((w) => w.name === n);
        const projectListW = getW("project_list");
        const projectNameW = getW("project_name");
        const promptW = getW("positive_text");
        const existingW = getW("existing_project");
        const idW = getW("project_id");
        const startPromptW = getW("start_prompt");
        const sceneW = getW("scene_number");

        hideInternalWidgets(this);

        this.refreshProjectList = async (preferredFile) => {
          try {
            const items = await fetchProjectFiles();
            const ctrl = projectListW;
            if (!ctrl) return;
            if (ctrl.options) ctrl.options.values = items;
            ctrl.values = items;
            let nextVal = ctrl.value;
            if (preferredFile && items.includes(preferredFile)) nextVal = preferredFile;
            else if (!items.includes(nextVal)) nextVal = items[0] || "none";
            ctrl.value = nextVal;
            if (typeof ctrl.callback === "function") ctrl.callback(nextVal, this, "project_list");
            this.onWidgetChanged?.(ctrl, nextVal, "project_list");
            this.setDirtyCanvas(true, true);
          } catch (e) {}
        };

        const doPartialLoad = async () => {
          const fileName = projectListW?.value;
          if (!fileName || fileName === "none") return;
          const currentScene = sceneW ? sceneW.value : 1;
          const safe = sanitizeFileName(fileName);
          try {
            const res = await fetch(`/onix/preview?file=${encodeURIComponent(safe)}&scene=${currentScene}`, {
              credentials: "same-origin",
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            if (idW) idW.value = data.id || "";
            if (projectNameW) projectNameW.value = data.name || safe.replace(/\.json$/i, "");
            const uiPrompt = Array.isArray(data.prompt_lines)
              ? data.prompt_lines.map((x) => (x == null ? "" : String(x))).join("\n")
              : data.prompt || "";
            if (promptW) promptW.value = uiPrompt;
            if (existingW) existingW.value = !!data.existing;
            if (startPromptW) startPromptW.value = Number.isFinite(data.start_prompt) ? data.start_prompt : 0;
            if (sceneW && Number.isFinite(data.scene_number)) sceneW.value = data.scene_number;
            setNameDisabled(projectNameW, !!data.existing);
            this.graph?.setDirtyCanvas?.(true, true);
          } catch (e) {
            console.warn("[OnixJS] preview failed:", e);
          }
        };

        const addBtn = (label, fn) => {
          const b = this.addWidget("button", label, null, () => fn());
          b.serialize = false;
          return b;
        };
        addBtn("Load Project", () => doPartialLoad());
        addBtn("New Project", () => {
          if (existingW) existingW.value = false;
          if (idW) idW.value = "";
          if (projectNameW) projectNameW.value = "";
          if (promptW) promptW.value = "";
          if (startPromptW) startPromptW.value = 0;
          setNameDisabled(projectNameW, false);
          this.graph?.setDirtyCanvas?.(true, true);
        });
        addBtn("Refresh List", () => this.refreshProjectList(projectListW?.value));

        hookAfterExec(this, (output) => {
          let obj = null;
          if (output && output.project && Array.isArray(output.project) && output.project.length) {
            obj = output.project[0];
          } else if (output && output.ui && typeof output.ui === "object") {
            const u = output.ui;
            if (u.project && Array.isArray(u.project) && u.project.length) obj = u.project[0];
          }
          if (!obj) return;
          if (existingW) existingW.value = true;
          if (idW && obj.id) idW.value = obj.id;
          this.refreshProjectList?.(obj.file || null);
          setNameDisabled(projectNameW, true);
          this.graph?.setDirtyCanvas?.(true, true);
        });

        return r;
      };
    }

    // Handling OnixExecutionTimer Node
    if (nodeData.name === "OnixExecutionTimer") {
      nodeType.prototype.onNodeCreated = function () {
        this.bgcolor = "#000000";
        this.color = "#000000";
        this.title = "Execution Timer";
        this.properties = this.properties || {};
        this.size = [420, 130];

        const container = document.createElement("div");
        container.style.cssText = `width: 100%; height: 100%; position: relative; --text-color: #7300ff; --glow-color: #7300ff;`;

        this.timerDisplay = document.createElement("div");
        this.timerDisplay.className = "onix-timer-display";
        this.timerDisplay.textContent = this.properties.elapsed_time_str || "00:00:000";
        
        container.appendChild(this.timerDisplay);
        this.addDOMWidget("onixTimer", "Onix Timer", container, { serialize: false });
        
        OnixGlobalTimer.registerNode(this);
      };
      
      nodeType.prototype.onRemoved = function() { OnixGlobalTimer.unregisterNode(this); };
      nodeType.prototype.onSerialize = function(o) { o.properties = this.properties; };
      nodeType.prototype.onConfigure = function(info) {
          this.properties = info.properties || {};
          if (this.timerDisplay) {
              this.timerDisplay.textContent = this.properties.elapsed_time_str || "00:00:000";
          }
      };
    }
  },

  setup() {
    // --- Styles for the Timer ---
    const style = document.createElement("style");
    style.innerText = `
        @keyframes onix-text-glow-pulse {
            0%, 100% { text-shadow: 0 0 25px var(--glow-color); }
            50% { text-shadow: 0 0 35px var(--glow-color); }
        }
        .onix-timer-display {
            text-align: center; width: 100%; height: 100%; position: absolute;
            top: 0; left: 0; background: transparent; border: none;
            color: var(--text-color);
            font-family: 'Courier New', 'Consolas', 'Monaco', monospace;
            box-sizing: border-box; outline: none; margin: 0;
            overflow: hidden; display: flex; justify-content: center;
            align-items: center; font-size: 50px;
            animation: onix-text-glow-pulse 10s infinite ease-in-out;
            transition: color 0.5s ease-in-out;
            font-variant-numeric: tabular-nums;
            letter-spacing: 0.1em;
            white-space: nowrap;
            font-weight: bold;
        }
    `;
    document.head.appendChild(style);

    // Load Orbitron font
    if (!document.querySelector('link[href*="Orbitron"]')) {
        const fontLink = document.createElement("link");
        fontLink.href = "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap";
        fontLink.rel = "stylesheet";
        document.head.appendChild(fontLink);
    }

    // ComfyUI API event listeners for global timer
    api.addEventListener("execution_start", () => OnixGlobalTimer.start());
    api.addEventListener("executing", ({ detail }) => {
        if (detail === null) OnixGlobalTimer.stop();
    });
    api.addEventListener("execution_error", () => OnixGlobalTimer.stop());
    api.addEventListener("execution_interrupted", () => OnixGlobalTimer.stop());
  }
});