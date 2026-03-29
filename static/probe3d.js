/**
 * MoCoLUS — 3D Probe Mesh Overlay (Three.js + STL)
 * Renders the probe_shell STL mesh with fan slice plane,
 * overlaid on top of the chest anatomy canvas.
 * Transparent background — chest map shows through.
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

class Probe3DViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.yaw = 0;
    this.pitch = 0;
    this.roll = 0;
    this.probeNx = 0.5;
    this.probeNy = 0.25;
    this.probeGroup = null;
    this._groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

    this._initScene();
    this._loadProbe();
    this._animate();
  }

  _initScene() {
    const w = this.container.clientWidth || 400;
    const h = this.container.clientHeight || 500;

    // Renderer — transparent background so chest canvas shows through
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x000000, 0); // fully transparent
    this.container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();

    // Camera — looking down at the probe from a 3/4 angle
    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(3.5, -4, 5.5);
    this.camera.lookAt(0, 0, 0.3);

    // Lighting
    this.scene.add(new THREE.AmbientLight(0x808898, 1.5));

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.8);
    dirLight.position.set(3, -2, 8);
    this.scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0x8898b0, 0.7);
    fillLight.position.set(-4, 3, 2);
    this.scene.add(fillLight);

    // Resize handler
    const ro = new ResizeObserver(() => this._onResize());
    ro.observe(this.container);
  }

  _buildFanSlice() {
    // Fan-shaped imaging slice beneath probe — dark translucent wedge
    // like the ultrasound beam footprint visible in the original UI
    const fanAngle = 55 * Math.PI / 180; // curvilinear fan angle
    const fanDepth = 2.2; // depth of fan in scene units
    const fanSegments = 24;

    const fanShape = new THREE.Shape();
    fanShape.moveTo(0, 0);
    for (let i = 0; i <= fanSegments; i++) {
      const a = -fanAngle / 2 + (fanAngle * i) / fanSegments;
      fanShape.lineTo(Math.sin(a) * fanDepth, -Math.cos(a) * fanDepth);
    }
    fanShape.lineTo(0, 0);

    const fanGeo = new THREE.ShapeGeometry(fanShape);
    const fanMat = new THREE.MeshBasicMaterial({
      color: 0x1a2030,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const fanMesh = new THREE.Mesh(fanGeo, fanMat);

    // Rotate fan so it extends downward from probe face (along -Z)
    fanMesh.rotation.x = Math.PI / 2;
    fanMesh.position.set(0, 0, -0.02); // just below probe face (z=0)

    // Fan border lines
    const borderMat = new THREE.LineBasicMaterial({
      color: 0x5a9ec0,
      transparent: true,
      opacity: 0.4,
    });
    const borderPts = [];
    borderPts.push(new THREE.Vector3(0, 0, -0.02));
    for (let i = 0; i <= fanSegments; i++) {
      const a = -fanAngle / 2 + (fanAngle * i) / fanSegments;
      borderPts.push(new THREE.Vector3(
        Math.sin(a) * fanDepth,
        0,
        -0.02 - Math.cos(a) * fanDepth
      ));
    }
    borderPts.push(new THREE.Vector3(0, 0, -0.02));
    const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPts);
    const borderLine = new THREE.Line(borderGeo, borderMat);

    // Depth lines inside fan
    const depthGroup = new THREE.Group();
    for (let d = 0.5; d < fanDepth; d += 0.5) {
      const arcPts = [];
      for (let i = 0; i <= 16; i++) {
        const a = -fanAngle / 2 + (fanAngle * i) / 16;
        arcPts.push(new THREE.Vector3(
          Math.sin(a) * d,
          0,
          -0.02 - Math.cos(a) * d
        ));
      }
      const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPts);
      const arcMat = new THREE.LineBasicMaterial({
        color: 0x3a5a70,
        transparent: true,
        opacity: 0.25,
      });
      depthGroup.add(new THREE.Line(arcGeo, arcMat));
    }

    return { fanMesh, borderLine, depthGroup };
  }

  _buildAxes() {
    // RGB axis arrows attached to probe group
    const axisGroup = new THREE.Group();
    const axisLen = 1.8;
    const configs = [
      { dir: [1, 0, 0], color: 0xff3333, rotZ: -Math.PI / 2 }, // X = Red
      { dir: [0, 1, 0], color: 0x33cc33, rotZ: 0 },             // Y = Green
      { dir: [0, 0, 1], color: 0x3388ff, rotX: Math.PI / 2 },   // Z = Blue
    ];

    for (const cfg of configs) {
      const shaftGeo = new THREE.CylinderGeometry(0.035, 0.035, axisLen, 10);
      const shaftMat = new THREE.MeshBasicMaterial({ color: cfg.color });
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      const d = new THREE.Vector3(...cfg.dir);
      shaft.position.copy(d.clone().multiplyScalar(axisLen / 2));
      if (cfg.rotZ) shaft.rotation.z = cfg.rotZ;
      if (cfg.rotX) shaft.rotation.x = cfg.rotX;
      axisGroup.add(shaft);

      const coneGeo = new THREE.ConeGeometry(0.09, 0.25, 10);
      const coneMat = new THREE.MeshBasicMaterial({ color: cfg.color });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      cone.position.copy(d.clone().multiplyScalar(axisLen + 0.12));
      if (cfg.rotZ) cone.rotation.z = cfg.rotZ;
      if (cfg.rotX) cone.rotation.x = cfg.rotX;
      axisGroup.add(cone);
    }

    return axisGroup;
  }

  _loadProbe() {
    const loader = new STLLoader();
    loader.load('/static/probe_mesh.stl', (geometry) => {
      // Same transforms as main_opengl.py
      geometry.computeBoundingBox();
      geometry.translate(0, 0, -geometry.boundingBox.min.z);

      geometry.scale(1, 1, -1);
      geometry.computeBoundingBox();
      geometry.translate(0, 0, -geometry.boundingBox.min.z);

      geometry.computeBoundingBox();
      const bb = geometry.boundingBox;
      geometry.translate(-(bb.min.x + bb.max.x) / 2, -(bb.min.y + bb.max.y) / 2, 0);

      // Rotate 90 CW in XY
      geometry.applyMatrix4(new THREE.Matrix4().set(
        0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ));

      geometry.scale(0.01, 0.01, 0.01);
      geometry.computeVertexNormals();

      // Semi-transparent orange (matches original)
      const material = new THREE.MeshPhongMaterial({
        color: 0xff8030,
        transparent: true,
        opacity: 0.75,
        shininess: 60,
        specular: 0x443322,
        side: THREE.DoubleSide,
      });

      const probeMesh = new THREE.Mesh(geometry, material);

      // Build probe group: mesh + fan + axes
      this.probeGroup = new THREE.Group();
      this.probeGroup.add(probeMesh);

      // Fan slice plane beneath probe
      const { fanMesh, borderLine, depthGroup } = this._buildFanSlice();
      this.probeGroup.add(fanMesh);
      this.probeGroup.add(borderLine);
      this.probeGroup.add(depthGroup);

      // Axes
      this.probeGroup.add(this._buildAxes());

      this.scene.add(this.probeGroup);
      this._updateOrientation();
    });
  }

  setOrientation(yaw, pitch, roll) {
    this.yaw = yaw;
    this.pitch = pitch;
    this.roll = roll;
    this._updateTransform();
  }

  /**
   * Set probe position in normalized chest coords [0,1].
   * Converts to screen NDC, raycasts to ground plane, and moves the 3D probe.
   */
  setPosition(nx, ny) {
    this.probeNx = nx;
    this.probeNy = ny;
    this._updateTransform();
  }

  _updateTransform() {
    if (!this.probeGroup) return;

    // --- Position: convert normalized chest coords to 3D world position ---
    // The chest canvas maps nx/ny [0,1] with padding:
    //   px = 22 + nx * (w - 44)  →  screenFracX = (22 + nx*(w-44)) / w
    //   py = 18 + ny * (h - 40)  →  screenFracY = (18 + ny*(h-40)) / h
    // Convert to NDC [-1, 1] for Three.js (Y is flipped)
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const screenX = 22 + this.probeNx * (w - 44);
    const screenY = 18 + this.probeNy * (h - 40);
    const ndcX = (screenX / w) * 2 - 1;
    const ndcY = -(screenY / h) * 2 + 1;

    // Raycast from camera through NDC point onto ground plane (z=0)
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const target = new THREE.Vector3();
    raycaster.ray.intersectPlane(this._groundPlane, target);

    if (target) {
      this.probeGroup.position.copy(target);
    }

    // --- Rotation ---
    const euler = new THREE.Euler(
      this.pitch * Math.PI / 180,
      this.roll * Math.PI / 180,
      this.yaw * Math.PI / 180,
      'ZXY'
    );
    this.probeGroup.setRotationFromEuler(euler);
  }

  resetOrientation() {
    this.yaw = 0;
    this.pitch = 0;
    this.roll = 0;
    this._updateTransform();
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w < 1 || h < 1) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.renderer.render(this.scene, this.camera);
  }
}

// Initialize when DOM ready
window.addEventListener('DOMContentLoaded', () => {
  window.probe3d = new Probe3DViewer('probe3d-container');
});
