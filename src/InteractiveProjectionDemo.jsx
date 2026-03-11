import { useRef, useState, useEffect, useMemo, useCallback, Suspense } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { motion, AnimatePresence } from 'framer-motion'
import { Cpu, TriangleAlert, Power } from 'lucide-react'
import * as THREE from 'three'

/* ─────────────────────────────────────────────
   1. Data helpers (pure functions, no retained state)
   ───────────────────────────────────────────── */

const BASE = import.meta.env.BASE_URL

async function loadBinPointCloud(url) {
  const resp = await fetch(url)
  const buf = await resp.arrayBuffer()
  const raw = new Float32Array(buf)
  const count = raw.length / 6
  const positions = new Float32Array(count * 3)
  const colors = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    positions[i * 3] = raw[i * 6]
    positions[i * 3 + 1] = -raw[i * 6 + 1]
    positions[i * 3 + 2] = raw[i * 6 + 2]
    colors[i * 3] = raw[i * 6 + 3]
    colors[i * 3 + 1] = raw[i * 6 + 4]
    colors[i * 3 + 2] = raw[i * 6 + 5]
  }
  return { positions, colors, count }
}

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
    vec3 normalized = (vWorldPos - uBoundsMin) / uBoundsRange;
    gl_FragColor = vec4(clamp(normalized, 0.0, 1.0), 1.0);
  }
`

/* ─────────────────────────────────────────────
   3. Inner R3F scene — dual-pass render-to-texture
   Runs inside <Canvas>, accesses gl context via useThree.
   ───────────────────────────────────────────── */

const RENDER_SIZE = 256

function PointCloudInner({ cloudData, onRGBTexture, onPointMapTexture }) {
  const { gl, camera, scene } = useThree()
  const pointsRef = useRef()
  const rgbMatRef = useRef()
  const pmMatRef = useRef()
  const rgbTargetRef = useRef(null)
  const pmTargetRef = useRef(null)
  const rgbCanvasRef = useRef(document.createElement('canvas'))
  const pmCanvasRef = useRef(document.createElement('canvas'))
  const pixelBufRef = useRef(new Uint8Array(RENDER_SIZE * RENDER_SIZE * 4))

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
    return {
      min: new THREE.Vector3(minX, minY, minZ),
      range: new THREE.Vector3(
        (maxX - minX) || 1,
        (maxY - minY) || 1,
        (maxZ - minZ) || 1,
      ),
    }
  }, [cloudData])

  // ── WebGLRenderTarget lifecycle ──
  useEffect(() => {
    const rtOpts = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, format: THREE.RGBAFormat }
    rgbTargetRef.current = new THREE.WebGLRenderTarget(RENDER_SIZE, RENDER_SIZE, rtOpts)
    pmTargetRef.current = new THREE.WebGLRenderTarget(RENDER_SIZE, RENDER_SIZE, { ...rtOpts })
    rgbCanvasRef.current.width = RENDER_SIZE
    rgbCanvasRef.current.height = RENDER_SIZE
    pmCanvasRef.current.width = RENDER_SIZE
    pmCanvasRef.current.height = RENDER_SIZE

    return () => {
      // Dispose render targets (frees GPU framebuffer + texture)
      rgbTargetRef.current?.dispose()
      pmTargetRef.current?.dispose()
      rgbTargetRef.current = null
      pmTargetRef.current = null
      // Dispose materials
      rgbMatRef.current?.dispose()
      pmMatRef.current?.dispose()
      rgbMatRef.current = null
      pmMatRef.current = null
      // Dispose geometry and its buffer attributes
      if (pointsRef.current) {
        const geom = pointsRef.current.geometry
        geom.dispose()
        pointsRef.current = null
      }
      // Sever refs to large typed arrays so V8 can GC them
      pixelBufRef.current = null
      rgbCanvasRef.current = null
      pmCanvasRef.current = null
    }
  }, [])

  // ── PointMap ShaderMaterial lifecycle ──
  useEffect(() => {
    const mat = new THREE.ShaderMaterial({
      vertexShader: pointMapVertexShader,
      fragmentShader: pointMapFragmentShader,
      uniforms: {
        uPointSize: { value: 2.0 },
        uBoundsMin: { value: bounds.min },
        uBoundsRange: { value: bounds.range },
      },
    })
    pmMatRef.current = mat
    return () => {
      mat.dispose()
      if (pmMatRef.current === mat) pmMatRef.current = null
    }
  }, [bounds])

  // ── Update geometry when data changes ──
  useEffect(() => {
    if (!pointsRef.current || !cloudData) return
    const geom = pointsRef.current.geometry
    // Dispose old buffer attributes before replacing
    const oldPos = geom.getAttribute('position')
    const oldCol = geom.getAttribute('color')
    geom.setAttribute('position', new THREE.BufferAttribute(cloudData.positions, 3))
    geom.setAttribute('color', new THREE.BufferAttribute(cloudData.colors, 3))
    geom.computeBoundingSphere()
    // Explicitly delete old attributes (severs reference to old Float32Arrays)
    if (oldPos) oldPos.array = null
    if (oldCol) oldCol.array = null
  }, [cloudData])

  // ── Throttled render-to-texture ──
  const frameCount = useRef(0)

  useFrame(() => {
    if (!pointsRef.current || !rgbTargetRef.current || !pmTargetRef.current || !pixelBufRef.current) return
    frameCount.current++
    if (frameCount.current % 3 !== 0) return

    const pts = pointsRef.current
    const origMat = pts.material

    // Pass 1: RGB
    gl.setRenderTarget(rgbTargetRef.current)
    gl.setClearColor(0x111827, 1)
    gl.clear()
    gl.render(scene, camera)

    gl.readRenderTargetPixels(rgbTargetRef.current, 0, 0, RENDER_SIZE, RENDER_SIZE, pixelBufRef.current)
    const rgbCtx = rgbCanvasRef.current.getContext('2d')
    const rgbImgData = rgbCtx.createImageData(RENDER_SIZE, RENDER_SIZE)
    for (let y = 0; y < RENDER_SIZE; y++) {
      const srcRow = (RENDER_SIZE - 1 - y) * RENDER_SIZE * 4
      const dstRow = y * RENDER_SIZE * 4
      rgbImgData.data.set(pixelBufRef.current.subarray(srcRow, srcRow + RENDER_SIZE * 4), dstRow)
    }
    rgbCtx.putImageData(rgbImgData, 0, 0)
    onRGBTexture(rgbCanvasRef.current.toDataURL())

    // Pass 2: PointMap
    pts.material = pmMatRef.current
    gl.setRenderTarget(pmTargetRef.current)
    gl.setClearColor(0x000000, 1)
    gl.clear()
    gl.render(scene, camera)
    pts.material = origMat

    gl.readRenderTargetPixels(pmTargetRef.current, 0, 0, RENDER_SIZE, RENDER_SIZE, pixelBufRef.current)
    const pmCtx = pmCanvasRef.current.getContext('2d')
    const pmImgData = pmCtx.createImageData(RENDER_SIZE, RENDER_SIZE)
    for (let y = 0; y < RENDER_SIZE; y++) {
      const srcRow = (RENDER_SIZE - 1 - y) * RENDER_SIZE * 4
      const dstRow = y * RENDER_SIZE * 4
      pmImgData.data.set(pixelBufRef.current.subarray(srcRow, srcRow + RENDER_SIZE * 4), dstRow)
    }
    pmCtx.putImageData(pmImgData, 0, 0)
    onPointMapTexture(pmCanvasRef.current.toDataURL())

    gl.setRenderTarget(null)
  })

  if (!cloudData) return null

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" array={cloudData.positions} count={cloudData.count} itemSize={3} />
        <bufferAttribute attach="attributes-color" array={cloudData.colors} count={cloudData.count} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial ref={rgbMatRef} size={0.003} vertexColors sizeAttenuation />
    </points>
  )
}

/* ─────────────────────────────────────────────
   4. Self-contained demo component
   ALL heavy state (cloudData, Canvas, refs) lives here.
   Unmounting this component destroys everything.
   ───────────────────────────────────────────── */

function PointCloudDemo({ onClose }) {
  const [timestep, setTimestep] = useState(1)
  const [rgbSrc, setRgbSrc] = useState(null)
  const [pmSrc, setPmSrc] = useState(null)
  const [cloudData, setCloudData] = useState(null)
  const [loading, setLoading] = useState(true)

  // Fetch .bin (or mock) — cancelled on unmount so no setState after destroy
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    const padded = String(timestep).padStart(2, '0')
    const url = `${BASE}pointclouds/frame_${padded}.bin`

    loadBinPointCloud(url)
      .then((data) => { if (!cancelled) { setCloudData(data); setLoading(false) } })
      .catch(() => { if (!cancelled) { setCloudData(generateMockPointCloud(timestep)); setLoading(false) } })

    return () => {
      cancelled = true
      // Sever reference to previous cloudData so the old Float32Arrays can be GC'd
      setCloudData(null)
    }
  }, [timestep])

  const handleRGB = useCallback((src) => setRgbSrc(src), [])
  const handlePM = useCallback((src) => setPmSrc(src), [])

  return (
    <>
      {/* Main container */}
      <div
        className="relative w-full rounded-2xl overflow-hidden border border-gray-700 bg-gray-950 touch-none select-none"
        style={{ aspectRatio: '16/9' }}
      >
        {/* 3D Canvas — destroyed entirely when PointCloudDemo unmounts */}
        <Canvas
          camera={{ position: [0.8, -0.3, 1.2], fov: 50, near: 0.01, far: 100 }}
          gl={{ antialias: true, preserveDrawingBuffer: true }}
          dpr={[1, 1.5]}
          onContextMenu={(e) => e.preventDefault()}
        >
          <ambientLight intensity={0.5} />
          <Suspense fallback={null}>
            <PointCloudInner
              cloudData={cloudData}
              onRGBTexture={handleRGB}
              onPointMapTexture={handlePM}
            />
          </Suspense>
          <OrbitControls
            target={[0, -0.1, 0.45]}
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

        {/* Close / Free Resources button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-50
                     bg-black/50 backdrop-blur-md border border-white/20 text-white
                     rounded-full px-4 py-2 flex items-center gap-2
                     hover:bg-red-500/80 hover:border-red-500
                     transition-all shadow-lg cursor-pointer
                     text-sm font-medium"
        >
          <Power className="w-4 h-4" />
          <span className="hidden sm:inline">Close Demo &amp; Free Memory</span>
          <span className="sm:hidden">Close Demo</span>
        </button>
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
        Drag to orbit the point cloud. The two viewports show the re-projected RGB image and the PointMap (XYZ &rarr; RGB encoding) from your exact camera angle &mdash; computed entirely on the GPU via WebGL render targets.
      </p>
    </>
  )
}

/* ─────────────────────────────────────────────
   5. Performance warning overlay
   ───────────────────────────────────────────── */

function PerformanceOverlay({ onActivate }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="relative w-full h-[540px] sm:h-[600px] rounded-2xl overflow-hidden border border-gray-700"
    >
      {/* Background preview image */}
      <img
        src={`${BASE}demo-preview.png`}
        alt=""
        className="absolute inset-0 w-full h-full object-cover"
      />
      {/* Dark scrim */}
      <div className="absolute inset-0 bg-gradient-to-br from-neutral-900/85 to-slate-900/90 backdrop-blur-sm" />

      {/* Glassmorphism card */}
      <div className="absolute inset-0 flex items-center justify-center p-6">
        <div className="backdrop-blur-md bg-black/40 border border-white/10 rounded-2xl p-8 max-w-md text-center shadow-2xl">
          {/* Icons */}
          <div className="flex items-center justify-center gap-3 mb-5">
            <div className="w-11 h-11 rounded-xl bg-amber-500/15 border border-amber-500/20 flex items-center justify-center">
              <TriangleAlert className="w-5 h-5 text-amber-400" />
            </div>
            <div className="w-11 h-11 rounded-xl bg-blue-500/15 border border-blue-500/20 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-blue-400" />
            </div>
          </div>

          {/* Title */}
          <h3 className="text-lg font-bold text-white mb-3">
            High-Performance WebGL Demo
          </h3>

          {/* Description */}
          <p className="text-sm text-neutral-300 leading-relaxed mb-6">
            This interactive point cloud reprojection requires a dedicated GPU.
            It may cause lag or drain battery on mobile devices or ultrabooks.
            <span className="block mt-1.5 text-neutral-500 text-xs">
              Approx. 10 MB data payload
            </span>
          </p>

          {/* Activate button */}
          <button
            onClick={onActivate}
            className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-full font-medium
                       transition-all duration-300 hover:scale-105
                       ring-4 ring-blue-500/30 hover:ring-blue-500/50
                       cursor-pointer shadow-lg shadow-blue-500/20"
          >
            Initialize Interactive Demo
          </button>
        </div>
      </div>
    </motion.div>
  )
}

/* ─────────────────────────────────────────────
   6. Main exported component (lightweight shell)
   Only holds isDemoActive boolean — no heavy data.
   ───────────────────────────────────────────── */

export default function InteractiveProjectionDemo() {
  const [isDemoActive, setIsDemoActive] = useState(false)

  return (
    <section id="demo" className="relative pt-16 pb-16 bg-gray-900">
      {/* Top gradient: white → dark */}
      <div className="absolute top-0 inset-x-0 h-24 bg-gradient-to-b from-white to-transparent pointer-events-none" />

      <div className="max-w-5xl mx-auto px-6">
        <h2 className="text-2xl font-bold text-white mb-2 pb-2 border-b border-gray-700">
          Interactive Reprojection Demo
        </h2>
        <p className="text-gray-400 text-sm mb-6">
          Orbit the 3D point cloud &mdash; two viewports render the projected RGB and PointMap in real time from your camera angle.
        </p>

        <AnimatePresence mode="wait">
          {isDemoActive ? (
            <motion.div
              key="demo"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
            >
              <PointCloudDemo onClose={() => setIsDemoActive(false)} />
            </motion.div>
          ) : (
            <PerformanceOverlay key="overlay" onActivate={() => setIsDemoActive(true)} />
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}
