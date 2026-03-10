import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { motion, useAnimation } from 'framer-motion'

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
   LAYOUT CONFIG
   ================================================================ */
const CARD_WIDTH = 320
const CARD_GAP = 24
const SPRING = { type: 'spring', stiffness: 300, damping: 30 }

/* ================================================================
   3D TRANSFORM HELPERS
   ================================================================ */
function getCardStyle(offset) {
  // offset: distance from center in card-units (0 = focused)
  const absOffset = Math.abs(offset)
  const sign = Math.sign(offset)

  if (absOffset < 0.01) {
    return { scale: 1.15, rotateY: 0, z: 50, opacity: 1, x: 0 }
  }

  const scale = Math.max(0.6, 1.15 - absOffset * 0.18)
  const rotateY = sign * Math.min(absOffset * 25, 45)
  const z = -absOffset * 80
  const opacity = Math.max(0.3, 1 - absOffset * 0.22)
  const x = 0

  return { scale, rotateY, z, opacity, x }
}

/* ================================================================
   SINGLE CARD
   ================================================================ */
function CoverflowCard({ filename, index, focusedIndex, onClick, videoRefs }) {
  const offset = index - focusedIndex
  const isFocused = Math.abs(offset) < 0.5
  const style = getCardStyle(offset)
  const base = import.meta.env.BASE_URL
  const title = parseTitle(filename)

  return (
    <motion.div
      className="flex-shrink-0 cursor-pointer select-none"
      style={{
        width: CARD_WIDTH,
        perspective: 1200,
        zIndex: 100 - Math.round(Math.abs(offset) * 10),
      }}
      onClick={() => onClick(index)}
      animate={{
        scale: style.scale,
        opacity: style.opacity,
        rotateY: style.rotateY,
        z: style.z,
      }}
      transition={SPRING}
    >
      <div
        className={`rounded-2xl overflow-hidden transition-shadow duration-500
          ${isFocused
            ? 'shadow-[0_20px_60px_-10px_rgba(59,130,246,0.3)]'
            : 'shadow-lg shadow-black/40'
          }`}
      >
        <div className="aspect-video bg-neutral-900 overflow-hidden">
          <video
            ref={(el) => { videoRefs.current[index] = el }}
            className="w-full h-full object-cover"
            src={`${base}videos/${filename}`}
            preload="metadata"
            muted
            loop
            playsInline
          />
        </div>
      </div>
      {/* Title — only visible on focused card */}
      <motion.p
        className="text-center mt-3 text-sm font-medium text-white/90 tracking-wide"
        animate={{ opacity: isFocused ? 1 : 0, y: isFocused ? 0 : 8 }}
        transition={{ duration: 0.3 }}
      >
        {title}
      </motion.p>
    </motion.div>
  )
}

/* ================================================================
   ARROW BUTTON
   ================================================================ */
function ArrowButton({ direction, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`absolute top-1/2 -translate-y-1/2 z-30
                  w-11 h-11 rounded-full flex items-center justify-center
                  bg-white/10 backdrop-blur-md border border-white/20
                  text-white/80 hover:bg-white/20 hover:text-white
                  transition-all duration-200 cursor-pointer
                  disabled:opacity-20 disabled:cursor-default
                  ${direction === 'left' ? 'left-4 sm:left-8' : 'right-4 sm:right-8'}`}
      aria-label={direction === 'left' ? 'Previous' : 'Next'}
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24">
        {direction === 'left'
          ? <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
          : <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
        }
      </svg>
    </button>
  )
}

/* ================================================================
   DOT INDICATORS
   ================================================================ */
function DotIndicators({ count, focusedIndex, onDotClick }) {
  return (
    <div className="flex justify-center gap-2 mt-8">
      {Array.from({ length: count }).map((_, i) => (
        <button
          key={i}
          onClick={() => onDotClick(i)}
          className={`w-2 h-2 rounded-full transition-all duration-300 cursor-pointer
            ${i === focusedIndex
              ? 'bg-blue-400 w-6'
              : 'bg-white/30 hover:bg-white/50'
            }`}
          aria-label={`Go to slide ${i + 1}`}
        />
      ))}
    </div>
  )
}

/* ================================================================
   MAIN COVERFLOW COMPONENT
   ================================================================ */
export default function CoolVideoCoverflow() {
  const [focusedIndex, setFocusedIndex] = useState(0)
  const videoRefs = useRef([])
  const containerRef = useRef(null)
  const controls = useAnimation()
  const dragStartX = useRef(0)
  const count = VIDEO_FILES.length

  // Calculate the x offset to center the focused card
  const getTargetX = useCallback((idx) => {
    // Each card occupies CARD_WIDTH + CARD_GAP
    const slot = CARD_WIDTH + CARD_GAP
    return -idx * slot
  }, [])

  // Animate track to center on focused card
  useEffect(() => {
    controls.start({ x: getTargetX(focusedIndex) }, SPRING)
  }, [focusedIndex, controls, getTargetX])

  // Video playback management: only focused video plays
  useEffect(() => {
    videoRefs.current.forEach((video, i) => {
      if (!video) return
      if (i === focusedIndex) {
        video.play().catch(() => {})
      } else {
        video.pause()
        video.currentTime = 0
      }
    })
  }, [focusedIndex])

  const goTo = useCallback((idx) => {
    setFocusedIndex(Math.max(0, Math.min(count - 1, idx)))
  }, [count])

  const handleDragStart = useCallback((_, info) => {
    dragStartX.current = info.point.x
  }, [])

  const handleDragEnd = useCallback((_, info) => {
    const dx = info.point.x - dragStartX.current
    const velocity = info.velocity.x
    const threshold = CARD_WIDTH / 4

    if (dx < -threshold || velocity < -300) {
      goTo(focusedIndex + 1)
    } else if (dx > threshold || velocity > 300) {
      goTo(focusedIndex - 1)
    } else {
      // snap back
      controls.start({ x: getTargetX(focusedIndex) }, SPRING)
    }
  }, [focusedIndex, goTo, controls, getTargetX])

  // Keyboard support
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowLeft') goTo(focusedIndex - 1)
      if (e.key === 'ArrowRight') goTo(focusedIndex + 1)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [focusedIndex, goTo])

  // Memoize video list to prevent re-renders
  const cards = useMemo(() => VIDEO_FILES.map((file, i) => (
    <CoverflowCard
      key={file}
      filename={file}
      index={i}
      focusedIndex={focusedIndex}
      onClick={goTo}
      videoRefs={videoRefs}
    />
  )), [focusedIndex, goTo])

  return (
    <section id="simulation" className="relative py-20 sm:py-28 bg-neutral-950 overflow-hidden">
      {/* Section heading */}
      <div className="max-w-5xl mx-auto px-6 mb-12 sm:mb-16">
        <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">
          Simulation Tasks
        </h2>
        <p className="text-neutral-400 text-sm sm:text-base">
          10 manipulation tasks from the RoboTwin 2.0 benchmark — drag or click to explore.
        </p>
      </div>

      {/* Coverflow viewport */}
      <div
        ref={containerRef}
        className="relative w-full overflow-hidden"
        style={{ perspective: 1200 }}
      >
        {/* Arrows */}
        <ArrowButton
          direction="left"
          onClick={() => goTo(focusedIndex - 1)}
          disabled={focusedIndex === 0}
        />
        <ArrowButton
          direction="right"
          onClick={() => goTo(focusedIndex + 1)}
          disabled={focusedIndex === count - 1}
        />

        {/* Gradient fade edges */}
        <div className="absolute inset-y-0 left-0 w-24 sm:w-40 bg-gradient-to-r from-neutral-950 to-transparent z-20 pointer-events-none" />
        <div className="absolute inset-y-0 right-0 w-24 sm:w-40 bg-gradient-to-l from-neutral-950 to-transparent z-20 pointer-events-none" />

        {/* Draggable track */}
        <div className="flex justify-center">
          <motion.div
            className="flex items-center"
            style={{
              gap: CARD_GAP,
              paddingLeft: '50%',
              paddingRight: '50%',
              marginLeft: -(CARD_WIDTH / 2),
              transformStyle: 'preserve-3d',
            }}
            drag="x"
            dragConstraints={{ left: -count * (CARD_WIDTH + CARD_GAP), right: 0 }}
            dragElastic={0.1}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            animate={controls}
          >
            {cards}
          </motion.div>
        </div>
      </div>

      {/* Dot indicators */}
      <DotIndicators count={count} focusedIndex={focusedIndex} onDotClick={goTo} />
    </section>
  )
}
