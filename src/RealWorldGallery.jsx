import { useState } from 'react'
import { motion } from 'framer-motion'

/* ================================================================
   VIDEO DATA
   ================================================================ */
const VIDEO_FILES = [
  'BeatBlock.mp4',
  'OpenDrawer.mp4',
  'PlaceCup.mp4',
  'Pouring.mp4',
  'StackCube.mp4',
  'SweepCup.mp4',
]

function parseTitle(filename) {
  const name = filename.replace(/\.mp4$/i, '')
  return name
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
}

/* ================================================================
   COMPONENT
   ================================================================ */
export default function RealWorldGallery() {
  const [hovered, setHovered] = useState(null)
  const base = import.meta.env.BASE_URL

  return (
    <section id="realworld" className="relative pt-10 pb-20 sm:pt-14 sm:pb-28 bg-gray-900">
      {/* Bottom gradient: dark -> white (into the Results section) */}
      <div className="absolute bottom-0 inset-x-0 h-24 bg-gradient-to-b from-transparent to-white pointer-events-none" />

      {/* Heading */}
      <div className="max-w-5xl mx-auto px-6 mb-10 sm:mb-14">
        <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">
          Real-World Experiments
        </h2>
        <p className="text-neutral-400 text-sm sm:text-base">
          6 manipulation tasks deployed on a dual-arm robot platform — hover to expand.
        </p>
      </div>

      {/* Accordion container */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 h-auto sm:h-[600px]">
          {VIDEO_FILES.map((file, i) => {
            const isHovered = hovered === i
            const someoneHovered = hovered !== null
            const title = parseTitle(file)

            return (
              <motion.div
                key={file}
                className="relative rounded-2xl overflow-hidden cursor-pointer h-[200px] sm:h-auto"
                style={{ flex: isHovered ? 5 : someoneHovered ? 0.6 : 1 }}
                animate={{ flex: isHovered ? 5 : someoneHovered ? 0.6 : 1 }}
                transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                {/* Video — autoPlay removed, use preload="auto" for reliability */}
                <video
                  className="absolute inset-0 w-full h-full object-cover"
                  src={`${base}videos_rw/${file}`}
                  autoPlay
                  muted
                  loop
                  playsInline
                  preload="auto"
                />

                {/* Dark gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent" />

                {/* Title — vertical when narrow, horizontal when expanded */}
                <div className="absolute inset-0 flex items-end">
                  <motion.div className="p-4 sm:p-5 w-full">
                    <motion.h3
                      className="font-semibold text-white whitespace-nowrap origin-bottom-left"
                      animate={{
                        rotate: isHovered ? 0 : -90,
                        x: isHovered ? 0 : -8,
                        y: isHovered ? 0 : -40,
                        fontSize: isHovered ? '1.25rem' : '0.875rem',
                      }}
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                    >
                      {title}
                    </motion.h3>

                    {/* Subtitle — only visible when expanded */}
                    <motion.p
                      className="text-white/60 text-sm mt-1"
                      animate={{
                        opacity: isHovered ? 1 : 0,
                        y: isHovered ? 0 : 10,
                      }}
                      transition={{ duration: 0.25, delay: isHovered ? 0.1 : 0 }}
                    >
                      Real-world deployment
                    </motion.p>
                  </motion.div>
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
