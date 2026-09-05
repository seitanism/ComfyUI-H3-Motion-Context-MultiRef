import { app } from "../../scripts/app.js";

// Keep saved widget order, but remove editable clocks from Update 9 examples.
const CLOCKS = new Set(["fps", "frame_rate", "source_fps", "start_fps", "end_fps", "force_rate"]);
function lockClocks(node) {
    if (!node?.properties?.h3_fixed_fps) return;
    for (const widget of node.widgets ?? []) {
        if (!CLOCKS.has(widget.name) || widget._h3FixedClock) continue;
        widget._h3FixedClock = true;
        widget.value = 24;
        widget.options ??= {};
        widget.options.min = 24;
        widget.options.max = 24;
        widget.disabled = true;
        const original = widget.beforeQueued;
        widget.beforeQueued = function (...args) {
            const result = original?.apply(this, args);
            this.value = 24;
            return result;
        };
    }
}
app.registerExtension({
    name: "seitanism.H3FixedFPS",
    nodeCreated(node) { queueMicrotask(() => lockClocks(node)); },
    afterConfigureGraph() {
        for (const node of app.graph?._nodes ?? []) lockClocks(node);
    },
});
