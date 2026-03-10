import { useState, useRef, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'

/* ================================================================
   VIDEO DATA
   ================================================================ */
const VIDEO_FILES = [
  'BeatBlockHammer.mp4',
  'BlocksRankingRGB.mp4',
  'ClickBell.mp4',
  'TurnSwitch.mp4',
  'PlaceContainerPlate.mp4',
  'PlaceEmptyCup.mp4',
  'StackBlocksTwo.mp4',
  'StackBlocksThree.mp4',
  'StackBowlsThree.mp4',
  'StackBowlsTwo.mp4',
]

function parseTitle(filename) {
  const name = filename.replace(/\.mp4$/i, '')
  return name
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
}

/* ================================================================
   CONFIG
   ================================================================ */
const CARD_W = 640
const CARD_SPACING = 280 // center-to-center distance for side cards (tighter overlap)
const SPRING = { type: 'spring', stiffness: 260, damping: 28 }

function getCardTransform(offset) {
  const abs = Math.abs(offset)
  const sign = Math.sign(offset)

  if (abs < 0.01) {
    return { scale: 1.0, rotateY: 0, opacity: 1 }
  }

  return {
    scale: Math.max(0.40, 0.66 - (abs - 1) * 0.1),
    rotateY: -sign * Math.min(abs * 30, 55),
    opacity: Math.max(0.2, 0.85 - abs * 0.2),
  }
}

/* ================================================================
   MAIN COMPONENT
   ================================================================ */
export default function CoolVideoCoverflow() {
  const [focused, setFocused] = useState(0)
  const videoRefs = useRef([])
  const dragRef = useRef({ startX: 0, dragging: false })
  const count = VIDEO_FILES.length
  const base = import.meta.env.BASE_URL

  const goTo = useCallback(
    (i) => setFocused(Math.max(0, Math.min(count - 1, i))),
    [count],
  )

  // Video playback: only focused plays
  useEffect(() => {
    videoRefs.current.forEach((v, i) => {
      if (!v) return
      if (i === focused) {
        v.play().catch(() => {})
      } else {
        v.pause()
        v.currentTime = 0
      }
    })
  }, [focused])

  // Keyboard
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'ArrowLeft') goTo(focused - 1)
      if (e.key === 'ArrowRight') goTo(focused + 1)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [focused, goTo])

  // Drag on the background only (not on buttons/cards)
  const onMouseDown = useCallback((e) => {
    // Only track drag if clicking the background area directly
    dragRef.current = { startX: e.clientX, dragging: false, startFocused: focused }
  }, [focused])

  const onMouseMove = useCallback((e) => {
    if (dragRef.current.startX === null) return
    if (Math.abs(e.clientX - dragRef.current.startX) > 10) {
      dragRef.current.dragging = true
    }
  }, [])

  const onMouseUp = useCallback((e) => {
    if (!dragRef.current.dragging) {
      dragRef.current.startX = null
      return
    }
    const dx = e.clientX - dragRef.current.startX
    const shift = Math.round(-dx / (CARD_SPACING * 0.6))
    goTo(dragRef.current.startFocused + shift)
    dragRef.current.startX = null
    dragRef.current.dragging = false
  }, [goTo])

  return (
    <section id="simulation" className="relative pt-20 pb-10 sm:pt-28 sm:pb-14 bg-gray-900 overflow-hidden select-none">
      {/* Top gradient: white -> dark */}
      <div className="absolute top-0 inset-x-0 h-24 bg-gradient-to-b from-white to-transparent pointer-events-none" />
      {/* Heading */}
      <div className="max-w-5xl mx-auto px-6 mb-12 sm:mb-16">
        <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">
          Simulation Tasks
        </h2>
        <p className="text-neutral-400 text-sm sm:text-base">
          10 manipulation tasks from the RoboTwin 2.0 benchmark — drag, click, or use arrows to explore.
        </p>
      </div>

      {/* Viewport */}
      <div
        className="relative w-full overflow-hidden"
        style={{ height: 440 }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {/* Gradient edges */}
        <div className="absolute inset-y-0 left-0 w-24 sm:w-44 bg-gradient-to-r from-gray-900 to-transparent z-20 pointer-events-none" />
        <div className="absolute inset-y-0 right-0 w-24 sm:w-44 bg-gradient-to-l from-gray-900 to-transparent z-20 pointer-events-none" />

        {/* Arrow buttons — z-40 above everything, pointer-events restored */}
        <button
          onClick={(e) => { e.stopPropagation(); goTo(focused - 1) }}
          disabled={focused === 0}
          className="absolute top-1/2 -translate-y-1/2 z-40 left-3 sm:left-8
                     w-12 h-12 rounded-full flex items-center justify-center
                     bg-white/10 backdrop-blur-md border border-white/20
                     text-white/80 hover:bg-white/25 hover:text-white hover:scale-110
                     transition-all duration-200 cursor-pointer
                     disabled:opacity-0 disabled:pointer-events-none"
          aria-label="Previous"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          </svg>
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); goTo(focused + 1) }}
          disabled={focused === count - 1}
          className="absolute top-1/2 -translate-y-1/2 z-40 right-3 sm:right-8
                     w-12 h-12 rounded-full flex items-center justify-center
                     bg-white/10 backdrop-blur-md border border-white/20
                     text-white/80 hover:bg-white/25 hover:text-white hover:scale-110
                     transition-all duration-200 cursor-pointer
                     disabled:opacity-0 disabled:pointer-events-none"
          aria-label="Next"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        </button>

        {/* Cards */}
        <div className="relative w-full h-full" style={{ perspective: 1200 }}>
          {VIDEO_FILES.map((file, i) => {
            const offset = i - focused
            const abs = Math.abs(offset)
            const t = getCardTransform(offset)

            // Focused card: centered at 0. Side cards: stacked closer together.
            const x = offset === 0 ? 0 : Math.sign(offset) * (CARD_W * 0.42 + (abs - 1) * CARD_SPACING + CARD_SPACING * 0.5)

            return (
              <motion.div
                key={file}
                className="absolute top-0 cursor-pointer"
                style={{
                  width: CARD_W,
                  left: '50%',
                  marginLeft: -(CARD_W / 2),
                  transformStyle: 'preserve-3d',
                  zIndex: 100 - Math.round(abs * 10),
                }}
                animate={{
                  x,
                  scale: t.scale,
                  rotateY: t.rotateY,
                  opacity: t.opacity,
                }}
                transition={SPRING}
                onClick={(e) => {
                  e.stopPropagation()
                  if (!dragRef.current.dragging) goTo(i)
                }}
              >
                <div
                  className={`rounded-2xl overflow-hidden transition-shadow duration-500
                    ${abs < 0.5
                      ? 'shadow-[0_20px_60px_-10px_rgba(59,130,246,0.4)]'
                      : 'shadow-lg shadow-black/60'
                    }`}
                >
                  <div className="aspect-video bg-neutral-900">
                    <video
                      ref={(el) => { videoRefs.current[i] = el }}
                      className="w-full h-full object-cover"
                      src={`${base}videos/${file}`}
                      preload="metadata"
                      muted
                      loop
                      playsInline
                    />
                  </div>
                </div>

                {/* Title — only on focused */}
                <motion.p
                  className="text-center mt-4 text-base font-medium text-white/90 tracking-wide pointer-events-none"
                  animate={{ opacity: abs < 0.5 ? 1 : 0, y: abs < 0.5 ? 0 : 10 }}
                  transition={{ duration: 0.25 }}
                >
                  {parseTitle(file)}
                </motion.p>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* Dots */}
      <div className="flex justify-center gap-2 mt-10">
        {VIDEO_FILES.map((_, i) => (
          <button
            key={i}
            onClick={() => goTo(i)}
            className={`h-2 rounded-full transition-all duration-300 cursor-pointer
              ${i === focused ? 'w-6 bg-blue-400' : 'w-2 bg-white/25 hover:bg-white/50'}`}
            aria-label={`Go to slide ${i + 1}`}
          />
        ))}
      </div>
    </section>
  )
}
