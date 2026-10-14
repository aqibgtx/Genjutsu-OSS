import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Use ComfyUI's supported upload endpoint; keep filenames, never base64, in the graph.
app.registerExtension({
  name: "Genjutsu.Uploads",
  nodeCreated(node) {
    if (node.comfyClass !== "GenjutsuTransform") return;
    const upload = (label, accept, multiple, field, category = null) => {
      node.addWidget("button", label, null, () => {
        const picker = document.createElement("input");
        picker.type = "file"; picker.accept = accept; picker.multiple = multiple;
        picker.onchange = async () => {
          try {
            const paths = [];
            for (const file of picker.files) {
              const data = new FormData();
              data.append("image", file); data.append("type", "input");
              data.append("subfolder", "genjutsu/uploads");
              const response = await api.fetchApi("/upload/image", { method: "POST", body: data });
              if (!response.ok) throw new Error(`Upload failed (${response.status})`);
              const result = await response.json();
              paths.push([result.subfolder, result.name].filter(Boolean).join("/"));
            }
            const widget = node.widgets.find((item) => item.name === field);
            if (category) {
              const previous = JSON.parse(widget.value || "[]");
              widget.value = JSON.stringify([...previous, ...paths.map((path) => ({ path, category, weight: 1 }))]);
            } else widget.value = multiple ? JSON.stringify(paths) : paths[0];
            widget.callback?.(widget.value);
            app.graph.setDirtyCanvas(true, true);
          } catch (error) { window.alert(error.message); }
        };
        picker.click();
      }, { serialize: false });
    };
    upload("Upload source video", "video/*", false, "source_video");
    upload("Upload references", "image/*", true, "reference_images_json");
    let rolesAdded = false;
    const addRoles = () => {
      if (rolesAdded || node.widgets.find((item) => item.name === "operation")?.value !== "full_recast") return;
      rolesAdded = true;
      upload("Add Character references", "image/*", true, "reference_images_json", "CHARACTER");
      upload("Add Environment references", "image/*", true, "reference_images_json", "LOCATION");
    };
    const configure = node.onConfigure;
    node.onConfigure = function () { const result = configure?.apply(this, arguments); addRoles(); return result; };
    const operation = node.widgets.find((item) => item.name === "operation");
    const changed = operation?.callback;
    if (operation) operation.callback = function () { const result = changed?.apply(this, arguments); addRoles(); return result; };
  },
});
