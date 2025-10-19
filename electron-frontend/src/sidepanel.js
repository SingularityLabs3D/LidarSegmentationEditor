export function initSidePanel(viewer) {
    const panel = document.createElement("div");
    panel.id = "object_sidepanel";
    panel.style.position = "absolute";
    panel.style.top = "0";
    panel.style.right = "0";
    panel.style.width = "220px";
    panel.style.height = "100%";
    panel.style.background = "rgba(0,0,0,0.6)";
    panel.style.color = "#eee";
    panel.style.zIndex = "200000";
    panel.style.fontFamily = "sans-serif";
    panel.style.padding = "10px";
    panel.style.overflowY = "auto";
    panel.style.borderLeft = "1px solid #333";

    const header = document.createElement("h3");
    header.textContent = "Objects";
    header.style.textAlign = "center";
    header.style.marginTop = "0";
    panel.appendChild(header);

    // mock data
    const objects = [
        { name: "Origin", position: [0, 0, 0] },
        { name: "Top of Pointcloud", position: [0, 0, 50] },
        { name: "Offset Example", position: [10, 5, 20] },
    ];

    const list = document.createElement("div");
    for (const obj of objects) {
        const row = document.createElement("div");
        row.style.marginBottom = "10px";
        row.style.borderBottom = "1px solid rgba(255,255,255,0.2)";
        row.style.paddingBottom = "6px";

        const label = document.createElement("div");
        label.textContent = obj.name;
        row.appendChild(label);

        const btn = document.createElement("button");
        btn.textContent = "Look at";
        btn.style.width = "100%";
        btn.style.marginTop = "4px";
        btn.style.cursor = "pointer";
        btn.onclick = () => lookAtPosition(obj.position);
        row.appendChild(btn);

        list.appendChild(row);
    }

    panel.appendChild(list);
    document.body.appendChild(panel);

    function lookAtPosition(coords) {
        if (!viewer?.scene?.view) {
            console.warn("Viewer not ready for lookAtPosition");
            return;
        }

        const view = viewer.scene.view;
        const target = new THREE.Vector3(...coords);

        const dir = view.position.clone().sub(view.getPivot()).normalize();
        const distance = view.position.distanceTo(view.getPivot());

        view.position.copy(target.clone().add(dir.multiplyScalar(distance)));
        view.lookAt(target);
        view.setView(view.position, target);

        viewer.scene.view = view;
        viewer.scene.viewChanged = true;
        viewer.postMessage(`Looking at ${coords.join(", ")}`, { duration: 1500 });
    }
}
