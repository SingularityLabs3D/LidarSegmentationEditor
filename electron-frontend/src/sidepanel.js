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

        const startPos = view.position.clone();
        const startTarget = view.getPivot().clone();

        const dir = startPos.clone().sub(startTarget).normalize();
        const distance = startPos.distanceTo(startTarget);

        const endTarget = target.clone();
        const endPos = target.clone().add(dir.multiplyScalar(distance));

        const from = { t: 0 };
        const to = { t: 1 };

        const posInterp = new THREE.Vector3();
        const targetInterp = new THREE.Vector3();

        const tween = new TWEEN.Tween(from)
            .to(to, 1000) // 1 second
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                posInterp.lerpVectors(startPos, endPos, from.t);
                targetInterp.lerpVectors(startTarget, endTarget, from.t);

                view.position.copy(posInterp);
                view.lookAt(targetInterp);
                view.setView(view.position, targetInterp);
                viewer.scene.viewChanged = true;
            })
            .onComplete(() => {
                view.setView(endPos, endTarget);
                viewer.scene.viewChanged = true;
                viewer.postMessage(`Looking at ${coords.join(", ")}`, { duration: 1500 });
            })
            .start();

        function animateTweens() {
            requestAnimationFrame(animateTweens);
            TWEEN.update();
        }
        animateTweens();
    }

}
