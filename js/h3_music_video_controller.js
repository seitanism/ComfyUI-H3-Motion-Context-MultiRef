import { app } from "../../scripts/app.js";

const NODE_NAME = "MiniMaxH3MusicVideoController";
const CONTROL_ID = "music_video";
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const MAX_CLIPS = 40;
const WATCHDOG_MS = 500;
let watchdog = null;
let reconciling = false;

function isController(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}

function widget(node, name, fallbackIndex = -1) {
    return node?.widgets?.find((w) => w.name === name) ??
        (fallbackIndex >= 0 ? node?.widgets?.[fallbackIndex] : null);
}

function groups(graph) {
    return graph?._groups ?? graph?.groups ?? [];
}

function controlMeta(group) {
    const meta = group?.flags?.h3_control;
    if (!meta || meta.controller !== CONTROL_ID) return null;
    return meta;
}

function centerInside(node, group) {
    const b = group?._bounding ?? group?.bounding;
    if (!b || b.length < 4 || !node?.pos) return false;
    const size = node.size ?? [0, 0];
    const cx = Number(node.pos[0]) + Number(size[0] ?? 0) / 2;
    const cy = Number(node.pos[1]) + Number(size[1] ?? 0) / 2;
    return cx >= b[0] && cx <= b[0] + b[2] && cy >= b[1] && cy <= b[1] + b[3];
}

function memberNodes(group, graph) {
    try { group.recomputeInsideNodes?.(); } catch (_) {}
    const out = new Set();
    const visit = (child) => {
        if (!child) return;
        if (typeof child.mode === "number") {
            out.add(child);
            return;
        }
        try { child.recomputeInsideNodes?.(); } catch (_) {}
        const kids = child?._children ? Array.from(child._children) : [];
        for (const item of kids) visit(item);
    };
    const children = group?._children ? Array.from(group._children) : [];
    for (const child of children) visit(child);
    if (!out.size) {
        for (const node of graph?._nodes ?? []) {
            if (centerInside(node, group)) out.add(node);
        }
    }
    return out;
}

function setNodeModeRecursive(node, mode, seen = new Set()) {
    if (!node || seen.has(node)) return;
    seen.add(node);
    const sub = node.subgraph;
    if (sub?._nodes) {
        for (const child of sub._nodes) setNodeModeRecursive(child, mode, seen);
    }
    if (node.mode !== mode) node.mode = mode;
}

function readState(controller) {
    return {
        active: Math.max(1, Math.min(MAX_CLIPS, Math.trunc(Number(widget(controller, "active_clips", 0)?.value ?? 1)))),
        previews: String(widget(controller, "previews", 1)?.value ?? "All Active"),
    };
}

function parameterNodes(graph) {
    return (graph?._nodes ?? []).filter(
        (node) => String(node?.properties?.h3_music_param ?? "") === "active_clips"
    );
}

function syncExecutionParameters(graph, state, { strict = false } = {}) {
    const params = parameterNodes(graph);
    const errors = [];
    if (params.length !== 1) {
        errors.push(`expected exactly one cache-isolated Music Video parameter 'active_clips', found ${params.length}`);
    } else {
        const w = widget(params[0], "value", 0);
        if (!w) {
            errors.push("Music Video active_clips parameter is missing widget 'value'");
        } else if (Number(w.value) !== Number(state.active)) {
            w.value = state.active;
            params[0].setDirtyCanvas?.(true, true);
        }
    }
    if (strict && errors.length) throw new Error("H3 Music Video Controller: " + errors.join("; "));
    return errors;
}

function desiredState(meta, state) {
    const role = String(meta.role ?? "");
    const index = Number(meta.index ?? 0);
    if (role === "music_clip") return index >= 1 && index <= state.active;
    if (role === "music_preview") {
        if (index < 1 || index > state.active) return false;
        if (state.previews === "All Active") return true;
        if (state.previews === "Last Active") return index === state.active;
        return false;
    }
    return null;
}

function validateManagedGroups(graph, managed, controller, state) {
    const errors = [];
    const allControllers = (graph?._nodes ?? []).filter(isController);
    if (allControllers.length !== 1) {
        errors.push(`expected exactly one H3 Music Video Controller, found ${allControllers.length}`);
    }
    for (const entry of managed) {
        if (Number(entry.meta.schema ?? 0) !== 1) {
            errors.push(`managed group ${entry.group?.title ?? "?"} has unsupported h3_control schema ${entry.meta.schema ?? "missing"}`);
        }
        if (!new Set(["music_clip", "music_preview"]).has(String(entry.meta.role ?? ""))) {
            errors.push(`managed group ${entry.group?.title ?? "?"} has unknown role ${entry.meta.role ?? "missing"}`);
        }
        const index = Number(entry.meta.index);
        if (!Number.isInteger(index) || index < 1 || index > MAX_CLIPS) {
            errors.push(`managed group ${entry.group?.title ?? "?"} has invalid clip index ${entry.meta.index ?? "missing"}`);
        }
        if (!entry.nodes.size) {
            errors.push(`managed group ${entry.group?.title ?? "?"} contains no nodes`);
        }
        if (entry.nodes.has(controller)) {
            errors.push(`the H3 Music Video Controller must not be inside managed group ${entry.group?.title ?? "?"}`);
        }
    }

    const indexes = managed
        .map((entry) => Number(entry.meta.index))
        .filter((index) => Number.isInteger(index) && index >= 1 && index <= MAX_CLIPS);
    const managedClipCount = indexes.length ? Math.max(...indexes) : 0;
    if (!managedClipCount) errors.push("expected at least one managed music clip and preview group");
    if (state.active > managedClipCount) {
        errors.push(`active_clips is ${state.active}, but the workflow contains only ${managedClipCount} managed clip slots`);
    }
    for (let i = 1; i <= managedClipCount; i++) {
        const clips = managed.filter((x) => x.meta.role === "music_clip" && Number(x.meta.index) === i);
        if (clips.length !== 1) errors.push(`expected exactly one music clip ${i} group, found ${clips.length}`);
        const previews = managed.filter((x) => x.meta.role === "music_preview" && Number(x.meta.index) === i);
        if (previews.length !== 1) errors.push(`expected exactly one music preview ${i} group, found ${previews.length}`);
    }

    const owners = new Map();
    for (const entry of managed) {
        for (const node of entry.nodes) {
            const list = owners.get(node) ?? [];
            list.push(entry);
            owners.set(node, list);
        }
    }
    for (const [node, entries] of owners) {
        if (entries.length > 1) {
            const labels = entries.map((e) => `${e.meta.role} ${e.meta.index ?? ""}`).join(", ");
            errors.push(`node ${node.id ?? "?"} (${node.title ?? node.type ?? "node"}) belongs to multiple H3-controlled groups: ${labels}`);
        }
    }
    return errors;
}

function isVHSVideoCombine(node) {
    return node?.comfyClass === "VHS_VideoCombine" || node?.type === "VHS_VideoCombine";
}

function bindVHSCompletedPreviewRefresh(node) {
    if (!isVHSVideoCombine(node) || node._h3CompletedPreviewRefreshBound) return;
    node._h3CompletedPreviewRefreshBound = true;

    const original = node.onExecuted;
    node.onExecuted = function(message, ...args) {
        const params = message?.gifs?.[0];
        const previewWidget = this.widgets?.find((w) => w.name === "videopreview");
        const current = previewWidget?.value?.params;

        // VHS currently skips updateParameters() when all returned file
        // parameters are unchanged, even when its onExecuted handler passes
        // force_update=true. This is common for temp previews that overwrite
        // the same filename on every run. Remember that condition before
        // calling VHS's own handler.
        const nativeWouldSkip = !!(
            params &&
            current &&
            !Object.entries(params).some(([key, value]) => current[key] !== value)
        );

        const result = original?.call(this, message, ...args);

        if (
            nativeWouldSkip &&
            params &&
            typeof this.updateParameters === "function" &&
            previewWidget?.value?.params
        ) {
            // Do not call updateSource() ourselves. Make exactly one parameter
            // differ, then let VHS's own forced update path reload the completed
            // video normally. The real filename is restored immediately by
            // updateParameters().
            delete previewWidget.value.params.filename;
            this.updateParameters(params, true);
        }
        return result;
    };
}

function bindManagedVHSPreviews(managed) {
    for (const entry of managed) {
        const role = String(entry.meta?.role ?? "");
        if (!role.includes("preview")) continue;
        for (const node of entry.nodes ?? []) {
            bindVHSCompletedPreviewRefresh(node);
        }
    }
}

function applyController(controller, { strict = false } = {}) {
    if (reconciling) return [];
    const graph = controller?.graph ?? app.graph;
    if (!graph) return [];
    reconciling = true;
    try {
        const state = readState(controller);
        const parameterErrors = syncExecutionParameters(graph, state, { strict });
        const managed = [];
        for (const group of groups(graph)) {
            const meta = controlMeta(group);
            if (!meta) continue;
            managed.push({ group, meta, nodes: memberNodes(group, graph) });
        }
        const errors = [...parameterErrors, ...validateManagedGroups(graph, managed, controller, state)];
        controller._h3ControllerErrors = errors;
        if (errors.length) {
            console.error("H3 Music Video Controller configuration error:\n" + errors.join("\n"));
            if (strict) throw new Error("H3 Music Video Controller: " + errors.join("; "));
            return errors;
        }
        bindManagedVHSPreviews(managed);
        for (const entry of managed) {
            const enabled = desiredState(entry.meta, state);
            if (enabled == null) continue;
            const mode = enabled ? MODE_ALWAYS : MODE_BYPASS;
            for (const node of entry.nodes) {
                if (node === controller) continue;
                setNodeModeRecursive(node, mode);
            }
        }
        graph.setDirtyCanvas?.(true, true);
        controller.setDirtyCanvas?.(true, true);
        return [];
    } finally {
        reconciling = false;
    }
}

function bindController(controller) {
    if (!controller || controller._h3MusicControllerBound) return;
    controller._h3MusicControllerBound = true;
    for (const w of controller.widgets ?? []) {
        const original = w.callback;
        w.callback = function(value, ...args) {
            const result = original?.call(this, value, ...args);
            queueMicrotask(() => applyController(controller));
            return result;
        };
        const originalBeforeQueued = w.beforeQueued;
        w.beforeQueued = function(...args) {
            applyController(controller, { strict: true });
            return originalBeforeQueued?.apply(this, args);
        };
    }
    queueMicrotask(() => applyController(controller));
}

function controllers(graph = app.graph) {
    return (graph?._nodes ?? []).filter(isController);
}

function reconcileAll(strict = false) {
    for (const node of controllers()) applyController(node, { strict });
}

app.registerExtension({
    name: "seitanism.H3MusicVideoController",
    async nodeCreated(node) {
        if (isController(node)) bindController(node);
    },
    async afterConfigureGraph() {
        for (const node of controllers()) bindController(node);
        queueMicrotask(() => reconcileAll(false));
    },
    async setup() {
        if (watchdog == null) {
            watchdog = setInterval(() => {
                if (!app.canvas?.isDragging && controllers().length) reconcileAll(false);
            }, WATCHDOG_MS);
        }
    },
});
