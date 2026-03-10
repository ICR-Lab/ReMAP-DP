import { useRef, useState, useEffect, useMemo, useCallback, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

/* ─────────────────────────────────────────────
   1. Data helpers
   ───────────────────────────────────────────── */

const BASE = import.meta.env.BASE_URL

// Load real .bin  (XYZRGB, 6 × float32 per point)
async function loadBinPointCloud(url) {
  const resp = await fetch(url)
  const buf = await resp.arrayBuffer()
  const raw = new Float32Array(buf)
  const count = raw.length / 6
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    positions[i * 3] = raw[i * 6]
    positions[i * 3 + 1] = raw[i * 6 + 1]
    positions[i * 3 + 2] = raw[i * 6 + 2]
    colors[i * 3] = raw[i * 6 + 3]
    colors[i * 3 + 1] = raw[i * 6 + 4]
    colors[i * 3 + 2] = raw[i * 6 + 5]
  }
  return { positions, colors, count }
}

// Generate synthetic fallback
function generateMockPointCloud(timestep) {
  const count = 30000
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  const t = timestep * 0.05
  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2
    const phi = Math.acos(2 * Math.random() - 1)
    const r = 0.3 + Math.random() * 0.4
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta) + Math.sin(t + i * 0.001) * 0.02
    positions[i * 3 + 1] = r * Math.cos(phi) + Math.cos(t * 1.3) * 0.01
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta) + 0.4
    colors[i * 3] = 0.3 + Math.random() * 0.4
    colors[i * 3 + 1] = 0.3 + Math.random() * 0.3
    colors[i * 3 + 2] = 0.2 + Math.random() * 0.3
  }
  return { positions, colors, count }
}

/* ─────────────────────────────────────────────
   2. Custom shader for PointMap pass
   ───────────────────────────────────────────── */

const pointMapVertexShader = /* glsl */ `
  varying vec3 vWorldPos;
  uniform float uPointSize;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
    gl_PointSize = uPointSize;
  }
`

const pointMapFragmentShader = /* glsl */ `
  varying vec3 vWorldPos;
  uniform vec3 uBoundsMin;
  uniform vec3 uBoundsRange;
  void main() {
    // Normalize world coords to [0,1] for RGB visualization
    vec3 normalized = (vWorldPos - uBoundsMin) / uBoundsRange;
    gl_FragColor = vec4(clamp(normalized, 0.0, 1.0), 1.0);
  }
`

/* ─────────────────────────────────────────────
   3. Points mesh with dual-pass rendering
   ───────────────────────────────────────────── */

const RENDER_SIZE = 256

function PointCloudScene({ cloudData, onRGBTexture, onPointMapTexture }) {
  const { gl, camera, scene } = useThree()
  const pointsRef = useRef()
  const rgbMatRef = useRef()
  const pmMatRef = useRef()
  const rgbTarget = useRef()
  const pmTarget = useRef()
  const rgbCanvas = useRef(document.createElement('canvas'))
  const pmCanvas = useRef(document.createElement('canvas'))
  const pixelBuf = useRef(new Uint8Array(RENDER_SIZE * RENDER_SIZE * 4))

  // Compute bounds for PointMap normalization
  const bounds = useMemo(() => {
    if (!cloudData) return { min: new THREE.Vector3(-1, -1, -1), range: new THREE.Vector3(2, 2, 2) }
    const pos = cloudData.positions
    let minX = Infinity, minY = Infinity, minZ = Infinity
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i] < minX) minX = pos[i]
      if (pos[i + 1] < minY) minY = pos[i + 1]
      if (pos[i + 2] < minZ) minZ = pos[i + 2]
      if (pos[i] > maxX) maxX = pos[i]
      if (pos[i + 1] > maxY) maxY = pos[i + 1]
      if (pos[i + 2] > maxZ) maxZ = pos[i + 2]
    }
    const rangeX = maxX - minX || 1
    const rangeY = maxY - minY || 1
    const rangeZ = maxZ - minZ || 1
    return {
      min: new THREE.Vector3(minX, minY, minZ),
      range: new THREE.Vector3(rangeX, rangeY, rangeZ),
    }
  }, [cloudData])

  // Create render targets
  useEffect(() => {
    rgbTarget.current = new THREE.WebGLRenderTarget(RENDER_SIZE, RENDER_SIZE, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    })
    pmTarget.current = new THREE.WebGLRenderTarget(RENDER_SIZE, RENDER_SIZE, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    })
    rgbCanvas.current.width = RENDER_SIZE
    rgbCanvas.current.height = RENDER_SIZE
    pmCanvas.current.width = RENDER_SIZE
    pmCanvas.current.height = RENDER_SIZE
    return () => {
      rgbTarget.current?.dispose()
      pmTarget.current?.dispose()
    }
  }, [])

  // Create PointMap shader material
  useEffect(() => {
    pmMatRef.current = new THREE.ShaderMaterial({
      vertexShader: pointMapVertexShader,
      fragmentShader: pointMapFragmentShader,
      uniforms: {
        uPointSize: { value: 2.0 },
        uBoundsMin: { value: bounds.min },
        uBoundsRange: { value: bounds.range },
      },
    })
    return () => pmMatRef.current?.dispose()
  }, [bounds])

  // Update geometry when data changes
  useEffect(() => {
    if (!pointsRef.current || !cloudData) return
    const geom = pointsRef.current.geometry
    geom.setAttribute('position', new THREE.BufferAttribute(cloudData.positions, 3))
    geom.setAttribute('color', new THREE.BufferAttribute(cloudData.colors, 3))
    geom.computeBoundingSphere()
  }, [cloudData])

  // Frame counter for throttled render-to-texture
  const frameCount = useRef(0)

  useFrame(() => {
    if (!pointsRef.current || !rgbTarget.current || !pmTarget.current) return
    frameCount.current++
    // Render to texture every 3 frames to save GPU
    if (frameCount.current % 3 !== 0) return

    const pts = pointsRef.current
    const origMat = pts.material

    // --- Pass 1: RGB ---
    gl.setRenderTarget(rgbTarget.current)
    gl.setClearColor(0x111827, 1) // bg-gray-900
    gl.clear()
    gl.render(scene, camera)

    // Read pixels for RGB overlay
    gl.readRenderTargetPixels(rgbTarget.current, 0, 0, RENDER_SIZE, RENDER_SIZE, pixelBuf.current)
    const rgbCtx = rgbCanvas.current.getContext('2d')
    const rgbImgData = rgbCtx.createImageData(RENDER_SIZE, RENDER_SIZE)
    // WebGL reads bottom-up, flip vertically
    for (let y = 0; y < RENDER_SIZE; y++) {
      const srcRow = (RENDER_SIZE - 1 - y) * RENDER_SIZE * 4
      const dstRow = y * RENDER_SIZE * 4
      rgbImgData.data.set(pixelBuf.current.subarray(srcRow, srcRow + RENDER_SIZE * 4), dstRow)
    }
    rgbCtx.putImageData(rgbImgData, 0, 0)
    onRGBTexture(rgbCanvas.current.toDataURL())

    // --- Pass 2: PointMap ---
    pts.material = pmMatRef.current
    gl.setRenderTarget(pmTarget.current)
    gl.setClearColor(0x000000, 1)
    gl.clear()
    gl.render(scene, camera)
    pts.material = origMat

    // Read pixels for PointMap overlay
    gl.readRenderTargetPixels(pmTarget.current, 0, 0, RENDER_SIZE, RENDER_SIZE, pixelBuf.current)
    const pmCtx = pmCanvas.current.getContext('2d')
    const pmImgData = pmCtx.createImageData(RENDER_SIZE, RENDER_SIZE)
    for (let y = 0; y < RENDER_SIZE; y++) {
      const srcRow = (RENDER_SIZE - 1 - y) * RENDER_SIZE * 4
      const dstRow = y * RENDER_SIZE * 4
      pmImgData.data.set(pixelBuf.current.subarray(srcRow, srcRow + RENDER_SIZE * 4), dstRow)
    }
    pmCtx.putImageData(pmImgData, 0, 0)
    onPointMapTexture(pmCanvas.current.toDataURL())

    // Restore render target
    gl.setRenderTarget(null)
  })

  if (!cloudData) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={cloudData.positions}
          count={cloudData.count}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          array={cloudData.colors}
          count={cloudData.count}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        ref={rgbMatRef}
        size={0.003}
        vertexColors
        sizeAttenuation
      />
    </points>
  )
}

/* ─────────────────────────────────────────────
   4. Main exported component
   ───────────────────────────────────────────── */

export default function InteractiveProjectionDemo() {
  const [timestep, setTimestep] = useState(1)
  const [rgbSrc, setRgbSrc] = useState(null)
  const [pmSrc, setPmSrc] = useState(null)
  const [cloudData, setCloudData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [availableBins, setAvailableBins] = useState([1]) // tracks which bins exist

  // Try loading real bin, fallback to mock
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    const padded = String(timestep).padStart(2, '0')
    const url = `${BASE}pointclouds/frame_${padded}.bin`

    loadBinPointCloud(url)
      .then((data) => {
        if (!cancelled) {
          setCloudData(data)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) {
          // Fallback to mock
          setCloudData(generateMockPointCloud(timestep))
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [timestep])

  const handleRGB = useCallback((src) => setRgbSrc(src), [])
  const handlePM = useCallback((src) => setPmSrc(src), [])

  return (
    <section id="demo" className="py-16 bg-gray-900">
      <div className="max-w-5xl mx-auto px-6">
        <h2 className="text-2xl font-bold text-white mb-2 pb-2 border-b border-gray-700">
          Interactive Reprojection Demo
        </h2>
        <p className="text-gray-400 text-sm mb-6">
          Orbit the 3D point cloud — two viewports render the projected RGB and PointMap in real time from your camera angle.
        </p>

        {/* Main container */}
        <div className="relative w-full rounded-2xl overflow-hidden border border-gray-700 bg-gray-950"
          style={{ aspectRatio: '16/9' }}
        >
          {/* 3D Canvas */}
          <Canvas
            camera={{ position: [0.8, 0.5, 1.0], fov: 50, near: 0.01, far: 100 }}
            gl={{ antialias: true, preserveDrawingBuffer: true }}
            dpr={[1, 1.5]}
          >
            <ambientLight intensity={0.5} />
            <Suspense fallback={null}>
              <PointCloudScene
                cloudData={cloudData}
                onRGBTexture={handleRGB}
                onPointMapTexture={handlePM}
              />
            </Suspense>
            <OrbitControls
              target={[0, 0.1, 0.45]}
              enableDamping
              dampingFactor={0.12}
              minDistance={0.3}
              maxDistance={3}
            />
          </Canvas>

          {/* Loading overlay */}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-20">
              <div className="flex items-center gap-3 text-gray-400 text-sm">
                <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Loading point cloud...
              </div>
            </div>
          )}

          {/* PIP overlays */}
          <div className="absolute bottom-3 right-3 flex gap-2 z-10">
            {/* RGB projection */}
            <div className="flex flex-col items-center">
              <div className="w-28 h-28 sm:w-36 sm:h-36 rounded-lg overflow-hidden border border-gray-600 bg-gray-900 shadow-lg">
                {rgbSrc ? (
                  <img src={rgbSrc} alt="RGB Projection" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">RGB</div>
                )}
              </div>
              <span className="text-[10px] text-gray-500 mt-1 font-medium">Projected RGB</span>
            </div>
            {/* PointMap projection */}
            <div className="flex flex-col items-center">
              <div className="w-28 h-28 sm:w-36 sm:h-36 rounded-lg overflow-hidden border border-gray-600 bg-gray-900 shadow-lg">
                {pmSrc ? (
                  <img src={pmSrc} alt="PointMap Projection" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">PointMap</div>
                )}
              </div>
              <span className="text-[10px] text-gray-500 mt-1 font-medium">Projected PointMap</span>
            </div>
          </div>

          {/* Camera angle label */}
          <div className="absolute top-3 left-3 z-10">
            <span className="px-2.5 py-1 rounded-md bg-black/50 backdrop-blur-sm text-[11px] text-gray-300 font-mono border border-gray-700">
              Orbit to reproject
            </span>
          </div>
        </div>

        {/* Timestep slider */}
        <div className="mt-5 flex items-center gap-4">
          <label className="text-sm text-gray-400 font-medium whitespace-nowrap">
            Time Step: <span className="text-white font-mono">{timestep}</span>
          </label>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={timestep}
            onChange={(e) => setTimestep(Number(e.target.value))}
            className="flex-1 h-1.5 rounded-full appearance-none bg-gray-700 accent-blue-500 cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:shadow-lg
                       [&::-webkit-slider-thumb]:cursor-pointer"
          />
          <div className="flex gap-1.5">
            {Array.from({ length: 10 }, (_, i) => i + 1).map((s) => (
              <button
                key={s}
                onClick={() => setTimestep(s)}
                className={`w-6 h-6 rounded text-[10px] font-mono transition-colors cursor-pointer
                  ${s === timestep
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-300'
                  }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <p className="text-xs text-gray-600 mt-3">
          Drag to orbit the point cloud. The two viewports show the re-projected RGB image and the PointMap (XYZ → RGB encoding) from your exact camera angle — computed entirely on the GPU via WebGL render targets.
        </p>
      </div>
    </section>
  )
}
