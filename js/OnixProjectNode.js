import { app } from "../../../scripts/app.js";

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

// Unified hook that wires node.onExecuted and socket "executed"
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

app.registerExtension({
  name: "OnixProjectNode",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "OnixProject") return;

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

      // Refresh dropdown and optionally select a file
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

          const changed = nextVal !== ctrl.value;
          ctrl.value = nextVal;

          if (typeof ctrl.callback === "function") ctrl.callback(nextVal, this, "project_list");
          this.onWidgetChanged?.(ctrl, nextVal, "project_list");

          this.setDirtyCanvas(true, true);
        } catch (e) {}
      };

      // Optional partial load helper
      const doPartialLoad = async () => {
        const fileName = projectListW?.value;
        if (!fileName || fileName === "none") return;
        
        // Pass current scene widget value to API to calculate start_prompt for THAT scene
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
          
          // API returns start_prompt specifically for the requested scene
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
        // Keep sceneW as is or reset? Usually keep to allow changing scene for new project
        setNameDisabled(projectNameW, false);
        this.graph?.setDirtyCanvas?.(true, true);
      });
      addBtn("Refresh List", () => this.refreshProjectList(projectListW?.value));

      const postBind = (file, pid, sceneNum) => {
        if (!file) return;
        if (existingW) existingW.value = true;
        if (idW && pid) idW.value = pid;
        // Optionally update scene widget if backend returned it (it should match input)
        // if (sceneW && sceneNum !== undefined) sceneW.value = sceneNum;

        this.refreshProjectList?.(file);
        setNameDisabled(projectNameW, true);
        this.graph?.setDirtyCanvas?.(true, true);
      };

      hookAfterExec(this, (output) => {
      let obj = null;
      if (output && output.project && Array.isArray(output.project) && output.project.length) {
        obj = output.project[0];
      } else if (output && output.ui && typeof output.ui === "object") {
        const u = output.ui;
        if (u.project && Array.isArray(u.project) && u.project.length) obj = u.project[0];
      }

      if (!obj) return;
      postBind(obj.file || null, obj.id || null, obj.scene_number);
    });

      return r;
    };
  },
});
