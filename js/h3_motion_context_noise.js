import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Reset button for H3 Motion Context Noise: deletes the latents it caught in catch_folder.
app.registerExtension({
    name: "h3.MotionContextNoise.reset",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MiniMaxH3MotionContextNoise") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.addWidget("button", "reset caught latents", null, async () => {
                const folder = this.widgets?.find((w) => w.name === "catch_folder")?.value ?? "";
                if (!confirm(`Delete every latent caught in "${folder}"?`)) return;
                try {
                    const res = await api.fetchApi("/h3_motion_context_noise/reset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ catch_folder: folder }),
                    });
                    const data = await res.json();
                    alert(data.error ? `Reset failed: ${data.error}` : `Deleted ${data.deleted} caught file(s).`);
                } catch (e) {
                    alert(`Reset failed: ${e}`);
                }
            }, { serialize: false });
            return r;
        };
    },
});
