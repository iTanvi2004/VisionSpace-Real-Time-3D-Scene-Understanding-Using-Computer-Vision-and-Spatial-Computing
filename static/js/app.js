/**
 * VisionSpace - Three.js Interactive 3D WebGL Scene & Real-Time Telemetry Client
 */

let scene, camera, renderer, controls;
let pointCloudMesh = null;
let boundingBoxGroup = null;
let currentPlyFilename = "";

document.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventListeners();
  // Start polling scene telemetry & 3D data every 250ms (4 Hz)
  setInterval(fetchSceneData, 250);
});

function initThreeJS() {
  const container = document.getElementById("threejs-container");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0f14);

  // Camera (Initial position elevated slightly behind origin)
  camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 50.0);
  camera.position.set(0, 2.2, -2.5);
  camera.lookAt(0, 0, 2.0);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0, 2.0);

  // Grid Floor
  const grid = new THREE.GridHelper(10, 20, 0x00d2ff, 0x223044);
  grid.position.y = -0.8;
  scene.add(grid);

  // Camera Origin Marker (Cyan cone pointing forward +Z)
  const camGeo = new THREE.ConeGeometry(0.12, 0.25, 4);
  camGeo.rotateX(Math.PI / 2);
  const camMat = new THREE.MeshBasicMaterial({ color: 0x00d2ff, wireframe: true });
  const camMesh = new THREE.Mesh(camGeo, camMat);
  camMesh.position.set(0, 0, 0);
  scene.add(camMesh);

  // Group for dynamic 3D bounding boxes
  boundingBoxGroup = new THREE.Group();
  scene.add(boundingBoxGroup);

  // Handle window resize
  window.addEventListener("resize", onWindowResize);

  // Animation Loop
  animate();
}

function onWindowResize() {
  const container = document.getElementById("threejs-container");
  if (!container || !renderer || !camera) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

async function fetchSceneData() {
  try {
    const res = await fetch("/api/scene_data");
    if (!res.ok) return;
    const data = await res.json();

    updateTelemetry(data);
    updatePointCloud(data.point_cloud);
    updateBoundingBoxes(data.spatial_objects);
    updateObjectsTable(data.spatial_objects);

    if (data.ply_filename) {
      currentPlyFilename = data.ply_filename;
    }
  } catch (err) {
    console.debug("Telemetry sync heartbeat:", err);
  }
}

function updateTelemetry(data) {
  // FPS
  const fpsEl = document.getElementById("fps-display");
  if (fpsEl && data.fps !== undefined) {
    fpsEl.textContent = `FPS: ${data.fps}`;
  }

  const sem = data.semantic_info || {};

  // Scene Type
  const sceneTypeEl = document.getElementById("scene-type");
  if (sceneTypeEl && sem.scene_type) {
    sceneTypeEl.textContent = sem.scene_type;
  }

  // Object Count
  const objCountEl = document.getElementById("object-count");
  if (objCountEl && sem.object_count !== undefined) {
    objCountEl.textContent = sem.object_count;
  }

  // Nearest Obstacle
  const nearestEl = document.getElementById("nearest-obstacle");
  if (nearestEl) {
    if (sem.nearest_object) {
      nearestEl.textContent = `${sem.nearest_object.class_name} (${sem.nearest_object.distance_m}m)`;
    } else {
      nearestEl.textContent = "Clear";
    }
  }

  // Hazard Alert Box
  const hazardBox = document.getElementById("hazard-box");
  const hazardText = document.getElementById("hazard-text");
  const hazardIcon = document.getElementById("hazard-icon");

  if (hazardBox && hazardText) {
    const hazards = sem.hazard_warnings || [];
    if (hazards.length > 0) {
      hazardBox.className = "hazard-alert critical";
      hazardText.textContent = hazards[0];
      if (hazardIcon) hazardIcon.setAttribute("data-lucide", "alert-triangle");
    } else {
      hazardBox.className = "hazard-alert safe";
      hazardText.textContent = "Path Clear - No Immediate Proximity Hazards";
      if (hazardIcon) hazardIcon.setAttribute("data-lucide", "shield-check");
    }
    if (window.lucide) lucide.createIcons();
  }

  // Narrative Summary
  const narrativeEl = document.getElementById("narrative-text");
  if (narrativeEl && sem.summary) {
    narrativeEl.textContent = sem.summary;
  }
}

function updatePointCloud(pcData) {
  if (!pcData || !pcData.points || pcData.points.length === 0) return;

  const positions = new Float32Array(pcData.points);
  const colors = new Float32Array(pcData.colors);

  if (!pointCloudMesh) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.035,
      vertexColors: true,
      transparent: true,
      opacity: 0.9
    });

    pointCloudMesh = new THREE.Points(geometry, material);
    scene.add(pointCloudMesh);
  } else {
    // Update existing BufferGeometry attributes
    const geometry = pointCloudMesh.geometry;
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }
}

function updateBoundingBoxes(objects) {
  if (!boundingBoxGroup) return;

  // Clear previous bounding box meshes
  while (boundingBoxGroup.children.length > 0) {
    const child = boundingBoxGroup.children[0];
    boundingBoxGroup.remove(child);
  }

  if (!objects || objects.length === 0) return;

  objects.forEach(obj => {
    const pos = obj.position_3d;
    const dims = obj.dimensions_3d;

    const w = Math.max(0.15, dims.width);
    const h = Math.max(0.15, dims.height);
    const d = Math.max(0.15, dims.depth);

    const geo = new THREE.BoxGeometry(w, h, d);

    let hexColor = 0x00e676; // Safe green
    if (obj.proximity_status === "CRITICAL") hexColor = 0xff3d71;
    else if (obj.proximity_status === "WARNING") hexColor = 0xffb300;

    const wireMat = new THREE.MeshBasicMaterial({ color: hexColor, wireframe: true });
    const boxMesh = new THREE.Mesh(geo, wireMat);

    // Three.js coordinates: Y is Up, so flip camera Y
    boxMesh.position.set(pos.x, -pos.y, pos.z);
    boundingBoxGroup.add(boxMesh);
  });
}

function updateObjectsTable(objects) {
  const tbody = document.getElementById("objects-table-body");
  if (!tbody) return;

  if (!objects || objects.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No objects detected in current scene</td></tr>`;
    return;
  }

  let html = "";
  objects.forEach(obj => {
    const statusClass = obj.proximity_status.toLowerCase();
    const pos = obj.position_3d;
    html += `
      <tr>
        <td><strong>${obj.class_name}</strong></td>
        <td>${obj.distance_m}m</td>
        <td>(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})</td>
        <td>${obj.bearing_deg > 0 ? "+" : ""}${obj.bearing_deg}&deg;</td>
        <td><span class="legend-item"><span class="color-dot ${statusClass === 'critical' ? 'red' : (statusClass === 'warning' ? 'yellow' : 'green')}"></span> ${obj.proximity_status}</span></td>
      </tr>
    `;
  });
  tbody.innerHTML = html;
}

function initEventListeners() {
  // Reset Camera View
  const resetBtn = document.getElementById("btn-reset-cam");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      camera.position.set(0, 2.2, -2.5);
      camera.lookAt(0, 0, 2.0);
      controls.target.set(0, 0, 2.0);
    });
  }

  // Export .PLY point cloud
  const exportBtn = document.getElementById("btn-export-ply");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      exportBtn.disabled = true;
      const originalHTML = exportBtn.innerHTML;
      exportBtn.innerHTML = `<i data-lucide="loader-2" class="spin"></i> Exporting...`;
      if (window.lucide) lucide.createIcons();

      // Trigger instant dynamic export & download
      window.location.href = "/api/export_ply";

      setTimeout(() => {
        exportBtn.innerHTML = `<i data-lucide="check"></i> Downloaded!`;
        if (window.lucide) lucide.createIcons();
        setTimeout(() => {
          exportBtn.innerHTML = originalHTML;
          exportBtn.disabled = false;
          if (window.lucide) lucide.createIcons();
        }, 2000);
      }, 1500);
    });
  }
}
