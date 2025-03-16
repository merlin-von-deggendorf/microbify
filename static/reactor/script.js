import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
const container = document.getElementById('canvas-container');
const staticPath = container.getAttribute('data-static-url');
// Create the scene
const scene = new THREE.Scene();

// Set up the camera with more distance
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 10,10); // move the camera back

// Create the renderer and add it to the document
const renderer = new THREE.WebGLRenderer({ antialias: true });
onWindowResize();

container.appendChild(renderer.domElement);

// Add orbit controls for interactive camera movement
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0,5, 0);
// Add a directional light to illuminate the model
let light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 10, 10);
light.target.position.set(0, 0, 0);
scene.add(light);
light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(0, 10, -10);
light.target.position.set(0, 0, 0);
scene.add(light);

// Load the MTL file and then the OBJ file using the materials
const mtlLoader = new MTLLoader();
mtlLoader.load(staticPath+"reactor/reaktor.mtl", (materials) => {
  materials.preload();
  const objLoader = new OBJLoader();
  objLoader.setMaterials(materials);
  objLoader.load(staticPath+"reactor/reaktor.obj", (object) => {
    scene.add(object);
  }, (xhr) => {
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
  }, (error) => {
    console.error('Error loading the OBJ model:', error);
  });
});

// Animation loop to render the scene continuously
function animate() {
  requestAnimationFrame(animate);
  controls.update(); // update controls for damping
  renderer.render(scene, camera);
}
animate();

function onWindowResize() {
  const newWidth = window.innerWidth * 0.5;
  const newHeight = window.innerHeight * 0.5;
  camera.aspect = newWidth / newHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(newWidth, newHeight);
}
window.addEventListener('resize', onWindowResize);