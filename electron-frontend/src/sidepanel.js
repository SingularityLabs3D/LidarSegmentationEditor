async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json(); // or res.text()
}


export function initSidePanel(viewer) {
    // === DOM setup ===
    const panel = document.createElement("div");
    panel.id = "object_sidepanel";
    Object.assign(panel.style, {
        position: "absolute",
        top: "0",
        right: "0",
        width: "250px",
        height: "100%",
        background: "rgba(0,0,0,0.6)",
        color: "#eee",
        zIndex: "200000",
        fontFamily: "sans-serif",
        padding: "10px",
        overflowY: "auto",
        borderLeft: "1px solid #333",
    });

    const header = document.createElement("h3");
    header.textContent = "Dynamic object detections";
    header.style.textAlign = "center";
    header.style.marginTop = "0";
    panel.appendChild(header);

    const list = document.createElement("div");
    list.id = "detection_list";
    panel.appendChild(list);

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "Delete selected objects";
    Object.assign(deleteBtn.style, {
        width: "100%",
        marginTop: "10px",
        padding: "6px",
        cursor: "pointer",
    });
    panel.appendChild(deleteBtn);

    document.body.appendChild(panel);

    deleteBtn.addEventListener("click", async () => {
      // collect selected ids from the side panel
      // const checked = Array.from(
      //   panel.querySelectorAll('#detection_list input[type="checkbox"]:checked')
      // ).map(cb => cb.dataset.id);

      // if (checked.length === 0) {
      //   alert("No objects selected!");
      //   return;
      // }

      deleteBtn.disabled = true;
      try {
        postJSON("http://127.0.0.1:8001/delete").then(console.log).catch(console.error);
      } catch (err) {
        console.error("Delete failed:", err);
        alert("Failed to delete selected objects.");
      } finally {
        deleteBtn.disabled = false;
      }
    });


    // === State ===
    let detections = [];

    // === Real API fetchers (FastAPI in another container) ===
    // Configure base URL via global, env-injected variable, or default Docker service DNS
    const API_BASE = window.DETECTION_API_BASE || "http://127.0.0.1:8001";

    /**
     * GET /detections?scene=<id>
     * Expected response: { detections: [{ id, type, coords:[x,y,z], zoomLevel }] }
     */
    async function fetchDetections(sceneId) {
        const url = new URL(`${API_BASE}/detections`);
        if (sceneId) url.searchParams.set("scene", sceneId);
        const res = await fetch(url.toString(), { headers: { accept: "application/json" } });
        if (!res.ok) throw new Error(`GET /detections failed: ${res.status}`);
        const data = await res.json();
        const arr = Array.isArray(data) ? data : (data.detections || []);
        // normalize + dedupe by id
        const seen = new Set();
        return arr
            .map(d => ({
                id: String(d.id),
                type: d.type || "unknown",
                coords: Array.isArray(d.coords) && d.coords.length === 3 ? d.coords : [0,0,0],
                zoomLevel: Number.isFinite(d.zoomLevel) ? d.zoomLevel : 2
            }))
            .filter(d => (seen.has(d.id) ? false : (seen.add(d.id), true)));
    }

    /**
     * POST /detections/delete
     * Body: { ids: [ "id1","id2", ... ] }
     */
    async function postDelete(ids) {
        const res = await fetch(`${API_BASE}/detections/delete`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ ids })
        });
        if (!res.ok) throw new Error(`POST /detections/delete failed: ${res.status}`);
        // optimistic update
        detections = detections.filter(d => !ids.includes(d.id));
        renderList();
    }

    // === Annotation utilities ===
    let annotationMap = new Map(); // id → annotation

    function clearAnnotations() {
        for (const ann of annotationMap.values()) {
            viewer.scene.annotations.remove(ann);
        }
        annotationMap.clear();
    }

    function syncAnnotations() {
        // remove stale
        for (const [id, ann] of annotationMap.entries()) {
            if (!detections.find(d => d.id === id)) {
                viewer.scene.annotations.remove(ann);
                annotationMap.delete(id);
            }
        }

        // add / update
        for (const det of detections) {
            let ann = annotationMap.get(det.id);

            if (!ann) {
                const title = $(`<span>${det.type}</span>`);
                ann = new Potree.Annotation({
                    position: new THREE.Vector3(...det.coords),
                    title: title,
                    // description: `(${det.coords.map(c => c.toFixed(2)).join(", ")})`,
                    cameraPosition: viewer.scene.view.position.clone(),
                    cameraTarget: new THREE.Vector3(...det.coords),
                });
                ann.scaleByDistance = true;
                ann.addEventListener("click", () => {
                    lookAtPosition(det.coords, det.zoomLevel);
                    let prevColor = document.getElementById(det.id).style.backgroundColor
                    document.getElementById(det.id).scrollIntoView();
                    document.getElementById(det.id).style.backgroundColor = "#144c32";
                    setTimeout(() => {
                        document.getElementById(det.id).style.backgroundColor = prevColor;
                    }, 5000)


                });
                viewer.scene.annotations.add(ann);
                annotationMap.set(det.id, ann);
            } else {
                // update existing position or label if needed
                ann.position.copy(new THREE.Vector3(...det.coords));
                // ann.description = `(${det.coords.map(c => c.toFixed(2)).join(", ")})`;
            }
        }

        viewer.scene.viewChanged = true;
    }

    // === Render list ===
    function renderList() {
        list.innerHTML = "";

        // --- Permanent Origin object ---
        const originRow = document.createElement("div");
        originRow.style.marginBottom = "10px";
        originRow.style.borderBottom = "1px solid rgba(255,255,255,0.3)";
        originRow.style.paddingBottom = "6px";

        const originLabel = document.createElement("div");
        originLabel.textContent = "Origin";
        originLabel.style.fontWeight = "bold";
        originRow.appendChild(originLabel);

        const originCoords = document.createElement("div");
        originCoords.textContent = "(0.00, 0.00, 0.00)";
        originCoords.style.color = "#aaa";
        originCoords.style.fontSize = "0.9em";
        originRow.appendChild(originCoords);

        const originBtn = document.createElement("button");
        originBtn.textContent = "Look at";
        originBtn.style.width = "100%";
        originBtn.style.marginTop = "4px";
        originBtn.onclick = () => lookAtPosition([0.0, 0.0, 0.0], 1);
        originRow.appendChild(originBtn);

        list.appendChild(originRow);
        // --------------------------------

        for (const det of detections) {
            const row = document.createElement("div");
            row.style.marginBottom = "10px";
            row.style.borderBottom = "1px solid rgba(255,255,255,0.2)";
            row.style.padding = "6px";
            row.id = det.id;

            const label = document.createElement("div");
            label.textContent = det.type;
            label.style.fontWeight = "bold";
            row.appendChild(label);

            const coords = document.createElement("div");
            coords.textContent = `(${det.coords.map((n) => n.toFixed(2)).join(", ")})`;
            coords.style.color = "#aaa";
            coords.style.fontSize = "0.9em";
            row.appendChild(coords);

            const btnRow = document.createElement("div");
            btnRow.style.display = "flex";
            btnRow.style.gap = "4px";
            btnRow.style.marginTop = "4px";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.dataset.id = det.id;

            const btn = document.createElement("button");
            btn.textContent = "Look at";
            btn.style.flex = "1";
            btn.onclick = () => lookAtPosition(det.coords, det.zoomLevel);

            btnRow.appendChild(checkbox);
            btnRow.appendChild(btn);
            row.appendChild(btnRow);

            list.appendChild(row);
        }

        syncAnnotations();
    }

    // === Look-at tween ===
    function lookAtPosition(coords, zoomLevel) {
        if (!viewer?.scene?.view) {
            console.warn("Viewer not ready for lookAtPosition");
            return;
        }

        const view = viewer.scene.view;
        const target = new THREE.Vector3(...coords);

        // configurable absolute params
        const yaw = Math.PI / 3;        // horizontal rotation (radians)
        const pitch = 0.4;            // vertical angle from horizon (radians)
        const duration = 1200;
        zoomLevel = zoomLevel ?? 1;

        // --- compute final camera position in spherical coords around target ---
        const offset = new THREE.Vector3();
        offset.x = zoomLevel * Math.cos(pitch) * Math.sin(yaw);
        offset.y = zoomLevel * Math.cos(pitch) * Math.cos(yaw);
        offset.z = zoomLevel * Math.sin(pitch);

        const endPos = target.clone().add(offset);

        // start and end values
        const startPos = view.position.clone();
        const startTarget = view.getPivot().clone();

        const from = { t: 0 };
        const to = { t: 1 };

        const posInterp = new THREE.Vector3();
        const targetInterp = new THREE.Vector3();

        const tween = new TWEEN.Tween(from)
            .to(to, duration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                posInterp.lerpVectors(startPos, endPos, from.t);
                targetInterp.lerpVectors(startTarget, target, from.t);

                view.position.copy(posInterp);
                view.lookAt(targetInterp);
                view.setView(view.position, targetInterp);
                viewer.scene.viewChanged = true;
            })
            .onComplete(() => {
                view.position.copy(endPos);
                view.lookAt(target);
                view.setView(endPos, target);
                viewer.scene.viewChanged = true;
                viewer.postMessage(
                    `Looking at ${coords.join(", ")} (zoom=${zoomLevel.toFixed(1)})`,
                    { duration: 1500 }
                );
            })
            .start();

        // run tweens each frame
        function animateTweens() {
            requestAnimationFrame(animateTweens);
            TWEEN.update();
        }
        animateTweens();
    }



    // === Hook delete button ===
    // deleteBtn.onclick = () => {
    //     const checked = Array.from(list.querySelectorAll("input[type=checkbox]:checked")).map(
    //         (cb) => cb.dataset.id
    //     );
    //     if (checked.length === 0) {
    //         alert("No objects selected!");
    //         return;
    //     }
    //     postDelete(checked);
    // };

    // === Polling loop ===
    async function pollLoop() {
        try {
            const newData = await fetchDetections();
            detections = newData;
            renderList();
        } catch (e) {
            console.error("Failed to fetch detections:", e);
        } finally {
            setTimeout(pollLoop, 5000); // every 5 seconds
        }
    }

    pollLoop();
}
