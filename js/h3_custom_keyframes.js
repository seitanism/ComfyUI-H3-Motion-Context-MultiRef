import { app } from "../../scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3CustomKeyframes",
    "MiniMaxH3CustomKeyframesMasked",
]);
const DEFAULT_POSITIONS = [1, 22, 79];
const MIN_KEYFRAMES = 1;
const MAX_KEYFRAMES = 32;

function stateWidget(node) {
    return node.widgets?.find((w) => w.name === "keyframe_state");
}

function readState(node) {
    const raw = stateWidget(node);
    let state = {
        count: 3,
        positions: [...DEFAULT_POSITIONS],
    };

    try {
        const parsed = JSON.parse(raw?.value || "");
        if (Number.isInteger(parsed?.count)) {
            state.count = Math.min(
                MAX_KEYFRAMES,
                Math.max(MIN_KEYFRAMES, parsed.count),
            );
        }
        if (Array.isArray(parsed?.positions)) {
            state.positions = parsed.positions.map(
                (v) => Math.trunc(Number(v)),
            );
        }
    } catch (_) {}

    while (state.positions.length < state.count) {
        const previous = state.positions.at(-1) ?? 1;
        state.positions.push(previous + 17);
    }

    state.positions = state.positions.slice(0, state.count);
    return state;
}

function hideStateWidget(node) {
    const widget = stateWidget(node);
    if (!widget || widget._h3CustomHidden) return;

    widget._h3CustomHidden = true;
    widget.computeSize = () => [0, -4];
}

function imageInputName(i) {
    return `keyframe_image_${i}`;
}

function positionWidgetName(i) {
    return `keyframe ${i} position`;
}

function findInput(node, name) {
    return node.inputs?.findIndex(
        (input) => input.name === name,
    ) ?? -1;
}

function ensureImageInput(node, i) {
    const name = imageInputName(i);
    if (findInput(node, name) >= 0) return;

    node.addInput(name, "IMAGE", {
        label: `keyframe ${i} image`,
    });
}

function removeImageInput(node, i) {
    const slot = findInput(node, imageInputName(i));
    if (slot < 0) return;

    if (node.inputs?.[slot]?.link != null) {
        node.disconnectInput(slot);
    }
    node.removeInput(slot);
}

function findPositionWidget(node, i) {
    return node.widgets?.find(
        (w) => w.name === positionWidgetName(i),
    );
}

function ensurePositionWidget(node, i, initialValue) {
    let widget = findPositionWidget(node, i);

    if (widget) {
        widget.value = initialValue;
        return widget;
    }

    widget = node.addWidget(
        "number",
        positionWidgetName(i),
        initialValue,
        (value) => {
            widget.value = Math.trunc(Number(value));
            writeState(node);
        },
        {
            min: 0,
            max: 99999,
            step: 1,
            precision: 0,
        },
    );

    // The hidden server-declared keyframe_state widget is the durable source
    // of truth for the position list.
    widget.serialize = false;
    widget.options ??= {};
    widget.options.serialize = false;
    return widget;
}

function removePositionWidget(node, i) {
    const widget = findPositionWidget(node, i);
    if (!widget || !node.widgets) return;

    const index = node.widgets.indexOf(widget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
    }
}

function writeState(node) {
    const raw = stateWidget(node);
    if (!raw) return;

    const positions = [];
    for (
        let i = 1;
        i <= node._h3CustomKeyframeCount;
        i++
    ) {
        positions.push(
            Math.trunc(
                Number(findPositionWidget(node, i)?.value ?? 1),
            ),
        );
    }

    raw.value = JSON.stringify({
        count: node._h3CustomKeyframeCount,
        positions,
    });
}

function ensureButtons(node) {
    if (
        node.widgets?.some(
            (w) => w.name === "+ Add keyframe",
        )
    ) {
        return;
    }

    const add = node.addWidget(
        "button",
        "+ Add keyframe",
        null,
        () => {
            if (
                node._h3CustomKeyframeCount >= MAX_KEYFRAMES
            ) {
                return;
            }

            const current = readState(node);
            const i = node._h3CustomKeyframeCount + 1;
            const previous =
                current.positions.at(-1) ?? 1;

            node._h3CustomKeyframeCount = i;
            ensureImageInput(node, i);
            ensurePositionWidget(
                node,
                i,
                previous + 17,
            );
            writeState(node);
            refreshNode(node);
        },
    );
    add.serialize = false;

    const remove = node.addWidget(
        "button",
        "- Remove keyframe",
        null,
        () => {
            if (
                node._h3CustomKeyframeCount <= MIN_KEYFRAMES
            ) {
                return;
            }

            const i = node._h3CustomKeyframeCount;
            removeImageInput(node, i);
            removePositionWidget(node, i);
            node._h3CustomKeyframeCount -= 1;
            writeState(node);
            refreshNode(node);
        },
    );
    remove.serialize = false;
}

function reorderWidgets(node) {
    if (!node.widgets) return;

    const raw = stateWidget(node);
    const normal = [];
    const positions = [];
    const buttons = [];

    for (const widget of node.widgets) {
        if (widget === raw) continue;

        if (/^keyframe \d+ position$/.test(widget.name)) {
            positions.push(widget);
        } else if (
            widget.name === "+ Add keyframe" ||
            widget.name === "- Remove keyframe"
        ) {
            buttons.push(widget);
        } else {
            normal.push(widget);
        }
    }

    positions.sort((a, b) => {
        const ai = Number(
            a.name.match(/\d+/)?.[0] ?? 0,
        );
        const bi = Number(
            b.name.match(/\d+/)?.[0] ?? 0,
        );
        return ai - bi;
    });

    node.widgets = [
        ...(raw ? [raw] : []),
        ...normal,
        ...positions,
        ...buttons,
    ];
}

function refreshNode(node) {
    reorderWidgets(node);

    const size = node.computeSize?.();
    if (size) {
        node.setSize(size);
    }

    app.graph?.setDirtyCanvas?.(true, true);
}

function buildUI(node) {
    hideStateWidget(node);

    const state = readState(node);
    node._h3CustomKeyframeCount = state.count;

    // Remove only stale dynamic slots beyond the serialized count.
    for (
        let i = MAX_KEYFRAMES;
        i > state.count;
        i--
    ) {
        removeImageInput(node, i);
        removePositionWidget(node, i);
    }

    for (let i = 1; i <= state.count; i++) {
        ensureImageInput(node, i);
        ensurePositionWidget(
            node,
            i,
            state.positions[i - 1],
        );
    }

    ensureButtons(node);
    writeState(node);
    refreshNode(node);
}

app.registerExtension({
    name: "seitanism.H3CustomKeyframes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;

        const originalCreated =
            nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result =
                originalCreated?.apply(this, arguments);
            setTimeout(() => buildUI(this), 0);
            return result;
        };

        const originalConfigure =
            nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result =
                originalConfigure?.apply(this, arguments);
            setTimeout(() => buildUI(this), 0);
            return result;
        };
    },
});
