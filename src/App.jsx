import { useState } from 'react'
import CoolVideoCoverflow from './CoolVideoCoverflow'
import RealWorldGallery from './RealWorldGallery'
import MethodPipeline from './MethodPipeline'
import InteractiveProjectionDemo from './InteractiveProjectionDemo'
import YouTubeDemo from './YouTubeDemo'
import VisitorTraffic from './VisitorTraffic'

const NAV_ITEMS = ['Abstract', 'Method', 'Demo', 'Simulation', 'Real-World', 'Results', 'BibTeX']

function Navbar() {
  const [open, setOpen] = useState(false)

  const scrollTo = (id) => {
    document.getElementById(id.toLowerCase().replace(/-/g, ''))?.scrollIntoView({ behavior: 'smooth' })
    setOpen(false)
  }

  return (
    <nav className="fixed top-0 inset-x-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
      <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
        <button onClick={() => scrollTo('hero')} className="font-semibold text-gray-900 tracking-tight text-lg cursor-pointer">
          ReMAP-DP
        </button>

        {/* Desktop */}
        <ul className="hidden md:flex gap-8 text-sm font-medium text-gray-600">
          {NAV_ITEMS.map((item) => (
            <li key={item}>
              <button
                onClick={() => scrollTo(item.toLowerCase())}
                className="hover:text-gray-900 transition-colors cursor-pointer"
              >
                {item}
              </button>
            </li>
          ))}
        </ul>

        {/* Mobile toggle */}
        <button
          className="md:hidden text-gray-700 cursor-pointer"
          onClick={() => setOpen(!open)}
          aria-label="Toggle menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            {open
              ? <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              : <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            }
          </svg>
        </button>
      </div>

      {/* Mobile menu */}
      {open && (
        <ul className="md:hidden bg-white border-t border-gray-100 px-6 pb-4 space-y-3 text-sm font-medium text-gray-600">
          {NAV_ITEMS.map((item) => (
            <li key={item}>
              <button
                onClick={() => scrollTo(item.toLowerCase())}
                className="block w-full text-left py-1 hover:text-gray-900 transition-colors cursor-pointer"
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      )}
    </nav>
  )
}

function SectionHeading({ children }) {
  return (
    <h2 className="text-2xl font-bold text-gray-900 mb-6 pb-2 border-b border-gray-200">
      {children}
    </h2>
  )
}

{/* ================================================================
    AUTHORS & AFFILIATIONS — edit here
    equalContribution: true  → adds * (equal contribution)
    corresponding: true      → adds † (corresponding author)
   ================================================================ */}
const AUTHORS = [
  // { name: 'First Last', superscript: '1' },
  // { name: 'XinZhang Yang', superscript: '1', corresponding: true, equalContribution: true  },
  // { name: 'Renjun Wu', superscript: '1', corresponding: true, equalContribution: true  },
  // { name: 'Hellen Ma', superscript: '2' , corresponding: true, equalContribution: true },
  // { name: 'Jinyan Liu', superscript: '1' , corresponding: true, equalContribution: true },
  // { name: 'Xuesong Li', superscript: '1', corresponding: true, equalContribution: true },
  {name: 'Anonymous Authors'}
]

const AFFILIATIONS = [
  // { id: '1', text: 'School of Computer Science and Technology, Beijing Institute of Technology' },
  // { id: '2', text: 'The Hong Kong University of Science and Technology (HKUST)'}
]
{/* ================================================================ */}

function Hero() {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background video */}
      <video
        className="absolute inset-0 w-full h-full object-cover"
        src={`${import.meta.env.BASE_URL}videos_rw/main_web.mp4`}
        autoPlay
        loop
        muted
        playsInline
      />
      {/* Dark overlay */}
      <div className="absolute inset-0 bg-black/60" />

      {/* Content */}
      <div className="relative z-10 text-center px-4 sm:px-6 max-w-4xl mx-auto">
        <h1 className="text-2xl sm:text-3xl md:text-5xl lg:text-6xl font-bold text-white leading-tight mb-4 sm:mb-6 drop-shadow-lg">
          ReMAP-DP: Reprojected Multi-view Aligned PointMaps for Diffusion Policy
        </h1>

        <p className="text-base text-white/60 mb-8">
          Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2026
        </p>

        {/* ---------- Author list ---------- */}
        <div className="flex flex-wrap justify-center gap-x-4 sm:gap-x-6 gap-y-2 text-base sm:text-lg text-white/90">
          {AUTHORS.map((a, i) => (
            <span key={i} className="font-medium">
              {a.name}
              {a.superscript && <sup className="text-xs ml-0.5">{a.superscript}</sup>}
              {a.equalContribution && <sup className="text-xs ml-0.5">*</sup>}
              {a.corresponding && <sup className="text-xs ml-0.5">&dagger;</sup>}
            </span>
          ))}
        </div>

        {/* ---------- Contribution footnotes ---------- */}
        {/* <p className="mt-2 text-xs text-white/40">
          * Equal contribution &nbsp;&nbsp; &dagger; Corresponding author
        </p> */}
        {/* <p className="mt-2 text-xs text-white/40">
            &dagger; Corresponding author
        </p> */}

        {/* ---------- Affiliations ---------- */}
        {AFFILIATIONS.length > 0 && (
          <div className="mt-3 space-y-0.5">
            {AFFILIATIONS.map((af) => (
              <p key={af.id} className="text-sm text-white/50">
                <sup>{af.id}</sup>{af.text}
              </p>
            ))}
          </div>
        )}

        {/* ---------- Glassmorphism buttons ---------- */}
        <div className="mt-8 sm:mt-10 flex flex-wrap justify-center gap-3 sm:gap-4">
          {/* Paper (arXiv) */}
          <a
            href="https://arxiv.org/abs/2603.14977"
            className="group inline-flex items-center gap-2.5 px-6 py-3 rounded-xl
                       bg-white/10 backdrop-blur-lg border border-white/20
                       text-white text-sm font-medium
                       shadow-lg shadow-black/10
                       hover:bg-white/20 hover:border-white/30 hover:scale-105 hover:shadow-xl
                       transition-all duration-300 ease-out cursor-pointer"
          >
            {/* Document icon */}
            <svg className="w-5 h-5 opacity-80 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
            Paper (arXiv)
          </a>

          {/* Code (GitHub) */}
          <a
            href="https://github.com/ICR-Lab/ReMAP-DP"
            className="group inline-flex items-center gap-2.5 px-6 py-3 rounded-xl
                       bg-white/10 backdrop-blur-lg border border-white/20
                       text-white text-sm font-medium
                       shadow-lg shadow-black/10
                       hover:bg-white/20 hover:border-white/30 hover:scale-105 hover:shadow-xl
                       transition-all duration-300 ease-out cursor-pointer"
          >
            {/* GitHub icon */}
            <svg className="w-5 h-5 opacity-80 group-hover:opacity-100 transition-opacity" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2Z" />
            </svg>
            Code (GitHub)
          </a>

          {/* Model */}
          <a
            href="https://huggingface.co/ICR-Lab/ReMAP-DP"
            className="group inline-flex items-center gap-2.5 px-6 py-3 rounded-xl
                       bg-white/10 backdrop-blur-lg border border-white/20
                       text-white text-sm font-medium
                       shadow-lg shadow-black/10
                       hover:bg-white/20 hover:border-white/30 hover:scale-105 hover:shadow-xl
                       transition-all duration-300 ease-out cursor-pointer"
          >
            {/* Database icon */}
            <svg className="w-5 h-5 opacity-80 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" strokeWidth={1.8} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
            </svg>
            Model
          </a>
        </div>
      </div>
    </section>
  )
}

function Abstract() {
  return (
    <section id="abstract" className="py-16">
      <div className="max-w-3xl mx-auto px-6">
        <SectionHeading>Abstract</SectionHeading>
        <p className="text-gray-700 leading-relaxed text-justify">
          Generalist robot policies built upon 2D visual representations excel at semantic reasoning but inherently lack the explicit 3D spatial awareness required for high-precision tasks. Existing 3D integration methods struggle to bridge this gap due to the structural irregularity of sparse point clouds and the geometric distortion introduced by multi-view orthographic rendering. To overcome these barriers, we present ReMAP-DP, a novel framework synergizing standardized perspective reprojection with a structure-aware dual-stream diffusion policy. By coupling the re-projected views with pixel-aligned PointMaps, our dual-stream architecture leverages learnable modality embeddings to fuse frozen semantic features and explicit geometric descriptors, ensuring precise implicit patch-level alignment. Extensive experiments across simulation and real-world environments demonstrate ReMAP-DP&apos;s superior performance in diverse manipulation tasks. On RoboTwin 2.0, it attains a 59.3% average success rate, outperforming the DP3 baseline by +6.6%. On ManiSkill 3, our method yields a 28% improvement over DP3 on the geometrically challenging Stack Cube task. Furthermore, ReMAP-DP exhibits remarkable real-world robustness, executing high-precision and dynamic manipulations with superior data efficiency from only a handful of demonstrations.
        </p>
      </div>
    </section>
  )
}

function Results() {
  return (
    <section id="results" className="py-16">
      <div className="max-w-4xl mx-auto px-6">
        <SectionHeading>Results</SectionHeading>

        <p className="text-gray-700 leading-relaxed mb-8 text-justify">
          We evaluate ReMAP-DP on two simulation benchmarks (RoboTwin 2.0 and ManiSkill 3) and six real-world manipulation tasks. All simulation results are averaged over 100 evaluation episodes per checkpoint.
        </p>

        {/* Table I: RoboTwin 2.0 */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">RoboTwin 2.0 Benchmark</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse whitespace-nowrap">
              <thead>
                <tr className="border-b-2 border-gray-300">
                  <th className="text-left py-3 px-3 font-semibold text-gray-900">Method</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Beat Block</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Blocks Rank</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Click Bell</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Turn Switch</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Place Cont.</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Place Cup</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Stack 2</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Stack 3</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Bowls 3</th>
                  <th className="text-center py-3 px-2 font-semibold text-gray-900">Bowls 2</th>
                  <th className="text-center py-3 px-3 font-semibold text-gray-900">Avg.</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                <tr className="border-b border-gray-100 bg-blue-50">
                  <td className="py-3 px-3 font-semibold">Ours</td>
                  <td className="text-center py-3 px-2">0.70</td>
                  <td className="text-center py-3 px-2 font-semibold">0.03</td>
                  <td className="text-center py-3 px-2 font-semibold">1.00</td>
                  <td className="text-center py-3 px-2 font-semibold">0.60</td>
                  <td className="text-center py-3 px-2">0.80</td>
                  <td className="text-center py-3 px-2 font-semibold">0.68</td>
                  <td className="text-center py-3 px-2 font-semibold">0.50</td>
                  <td className="text-center py-3 px-2 font-semibold">0.08</td>
                  <td className="text-center py-3 px-2 font-semibold">0.72</td>
                  <td className="text-center py-3 px-2">0.82</td>
                  <td className="text-center py-3 px-3 font-semibold">59.3%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-3">DP3</td>
                  <td className="text-center py-3 px-2 font-semibold">0.72</td>
                  <td className="text-center py-3 px-2 font-semibold">0.03</td>
                  <td className="text-center py-3 px-2">0.90</td>
                  <td className="text-center py-3 px-2">0.46</td>
                  <td className="text-center py-3 px-2 font-semibold">0.86</td>
                  <td className="text-center py-3 px-2">0.65</td>
                  <td className="text-center py-3 px-2">0.24</td>
                  <td className="text-center py-3 px-2">0.01</td>
                  <td className="text-center py-3 px-2">0.57</td>
                  <td className="text-center py-3 px-2 font-semibold">0.83</td>
                  <td className="text-center py-3 px-3">52.7%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-3">ACT</td>
                  <td className="text-center py-3 px-2">0.56</td>
                  <td className="text-center py-3 px-2">0.01</td>
                  <td className="text-center py-3 px-2">0.58</td>
                  <td className="text-center py-3 px-2">0.05</td>
                  <td className="text-center py-3 px-2">0.72</td>
                  <td className="text-center py-3 px-2">0.61</td>
                  <td className="text-center py-3 px-2">0.25</td>
                  <td className="text-center py-3 px-2">0.00</td>
                  <td className="text-center py-3 px-2">0.48</td>
                  <td className="text-center py-3 px-2">0.82</td>
                  <td className="text-center py-3 px-3">40.8%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-3">DP</td>
                  <td className="text-center py-3 px-2">0.42</td>
                  <td className="text-center py-3 px-2">0.00</td>
                  <td className="text-center py-3 px-2">0.54</td>
                  <td className="text-center py-3 px-2">0.36</td>
                  <td className="text-center py-3 px-2">0.41</td>
                  <td className="text-center py-3 px-2">0.37</td>
                  <td className="text-center py-3 px-2">0.07</td>
                  <td className="text-center py-3 px-2">0.00</td>
                  <td className="text-center py-3 px-2">0.63</td>
                  <td className="text-center py-3 px-2">0.61</td>
                  <td className="text-center py-3 px-3">34.1%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mt-3">
            Table I: Comparison on RoboTwin 2.0. ReMAP-DP achieves the best average success rate of 59.3%, outperforming DP3 by +6.6%.
          </p>
        </div>

        {/* Table II: ManiSkill 3 */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">ManiSkill 3 Benchmark</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b-2 border-gray-300">
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Method</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Push Cube</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Stack Cube</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Pick Cube</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Lift Peg</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Peg Insert</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Avg.</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                <tr className="border-b border-gray-100 bg-blue-50">
                  <td className="py-3 px-4 font-semibold">Ours</td>
                  <td className="text-center py-3 px-4 font-semibold">0.95</td>
                  <td className="text-center py-3 px-4 font-semibold">0.28</td>
                  <td className="text-center py-3 px-4 font-semibold">0.48</td>
                  <td className="text-center py-3 px-4">0.36</td>
                  <td className="text-center py-3 px-4">0.01</td>
                  <td className="text-center py-3 px-4 font-semibold">41.6%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4">DP</td>
                  <td className="text-center py-3 px-4">0.92</td>
                  <td className="text-center py-3 px-4">0.01</td>
                  <td className="text-center py-3 px-4">0.24</td>
                  <td className="text-center py-3 px-4 font-semibold">0.75</td>
                  <td className="text-center py-3 px-4">0.01</td>
                  <td className="text-center py-3 px-4">38.6%</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4">DP3</td>
                  <td className="text-center py-3 px-4">0.88</td>
                  <td className="text-center py-3 px-4">0.00</td>
                  <td className="text-center py-3 px-4">0.44</td>
                  <td className="text-center py-3 px-4">0.03</td>
                  <td className="text-center py-3 px-4">0.01</td>
                  <td className="text-center py-3 px-4">27.2%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mt-3">
            Table II: Success rates on ManiSkill 3. ReMAP-DP achieves a 28% success rate on the geometrically challenging Stack Cube task, significantly outperforming DP3 (1%) and DP (0%).
          </p>
        </div>

        {/* Table III: Real-World */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Real-World Experiments</h3>
          <p className="text-gray-700 leading-relaxed mb-4 text-justify">
            We deployed ReMAP-DP on a Discover Airbot dual-arm platform (6-DOF per arm) with three Intel RealSense cameras. For each of 6 tasks, we collected 50 demonstrations and ran 10 evaluation trials with randomized object perturbations.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b-2 border-gray-300">
                  <th className="text-left py-3 px-4 font-semibold text-gray-900">Task</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">Ours</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">DP</th>
                  <th className="text-center py-3 px-4 font-semibold text-gray-900">DP3</th>
                </tr>
              </thead>
              <tbody className="text-gray-700">
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Beat w. Hammer</td>
                  <td className="text-center py-3 px-4 font-semibold">0.5</td>
                  <td className="text-center py-3 px-4">0.4</td>
                  <td className="text-center py-3 px-4">0.2</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Sweep Table</td>
                  <td className="text-center py-3 px-4 font-semibold">0.5</td>
                  <td className="text-center py-3 px-4">0.3</td>
                  <td className="text-center py-3 px-4">0.1</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Open Drawer PnP</td>
                  <td className="text-center py-3 px-4 font-semibold">0.3</td>
                  <td className="text-center py-3 px-4">0.1</td>
                  <td className="text-center py-3 px-4">0.0</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Place Empty Cup</td>
                  <td className="text-center py-3 px-4 font-semibold">0.6</td>
                  <td className="text-center py-3 px-4 font-semibold">0.6</td>
                  <td className="text-center py-3 px-4">0.3</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Pouring</td>
                  <td className="text-center py-3 px-4 font-semibold">0.6</td>
                  <td className="text-center py-3 px-4">0.4</td>
                  <td className="text-center py-3 px-4">0.1</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 px-4 italic">Stack Cube</td>
                  <td className="text-center py-3 px-4 font-semibold">0.3</td>
                  <td className="text-center py-3 px-4">0.1</td>
                  <td className="text-center py-3 px-4">0.0</td>
                </tr>
                <tr className="border-t-2 border-gray-300 bg-blue-50">
                  <td className="py-3 px-4 font-semibold">Average</td>
                  <td className="text-center py-3 px-4 font-semibold">46.7%</td>
                  <td className="text-center py-3 px-4">31.7%</td>
                  <td className="text-center py-3 px-4">11.7%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mt-3">
            Table III: Real-world success rates across 10 trials. ReMAP-DP achieves 46.7% average, substantially outperforming DP (31.7%) and DP3 (11.7%).
          </p>
        </div>

        {/* Key Findings */}
        <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Key Findings</h3>
          <ul className="space-y-2 text-gray-700 text-sm list-disc list-inside">
            <li>On RoboTwin 2.0, ReMAP-DP achieves <strong>59.3%</strong> average success rate, outperforming DP3 by <strong>+6.6%</strong> and standard DP by <strong>+25.2%</strong>.</li>
            <li>On ManiSkill 3 Stack Cube, ReMAP-DP reaches <strong>28%</strong> success vs. 1% for DP3, demonstrating the efficacy of dual-stream geometric alignment.</li>
            <li>In real-world deployment, ReMAP-DP achieves <strong>46.7%</strong> average with 50 demonstrations per task, nearly <strong>4&times;</strong> the success rate of DP3.</li>
            <li>Ablation studies confirm that dense PointMap ViT encoding (69% vs. &lt;5% for sparse), multi-view projection (+8%), and modality embeddings (+2%) all contribute to performance.</li>
          </ul>
        </div>
      </div>
    </section>
  )
}

function BibTeX() {
  const [copied, setCopied] = useState(false)

  const bibtex = `@misc{yang2026remapdpreprojectedmultiviewaligned,
      title={ReMAP-DP: Reprojected Multi-view Aligned PointMaps for Diffusion Policy}, 
      author={Xinzhang Yang and Renjun Wu and Jinyan Liu and Xuesong Li},
      year={2026},
      eprint={2603.14977},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.14977}, 
}`

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section id="bibtex" className="py-16 bg-gray-50">
      <div className="max-w-3xl mx-auto px-6">
        <SectionHeading>BibTeX</SectionHeading>
        <p className="text-gray-700 mb-4">If you find this work useful, please cite:</p>
        <div className="relative">
          <pre className="bg-gray-900 text-gray-100 rounded-xl p-4 sm:p-6 text-xs sm:text-sm overflow-x-auto leading-relaxed">
            {bibtex}
          </pre>
          <div className="absolute top-3 right-3">
            <button
              onClick={handleCopy}
              className="relative px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs rounded-md transition-colors cursor-pointer"
            >
              {/* Floating tooltip */}
              <span
                className={`absolute -top-9 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-green-600 text-white text-xs rounded-md whitespace-nowrap
                            pointer-events-none transition-all duration-300
                            ${copied ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1'}
                            after:content-[''] after:absolute after:top-full after:left-1/2 after:-translate-x-1/2
                            after:border-4 after:border-transparent after:border-t-green-600`}
              >
                Copied!
              </span>
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="py-8 border-t border-gray-200">
      <div className="max-w-5xl mx-auto px-6 text-center text-sm text-gray-400">
        <p>&copy; 2025 ReMAP-DP</p>
      </div>
    </footer>
  )
}

function App() {
  return (
    <div className="min-h-screen bg-white text-gray-800 font-sans antialiased">
      <Navbar />
      <main>
        <Hero />
        <Abstract />
        <YouTubeDemo />
        <MethodPipeline />
        <InteractiveProjectionDemo />
        <CoolVideoCoverflow />
        <RealWorldGallery />
        <Results />
        <BibTeX />
      </main>
      <VisitorTraffic />
      <Footer />
    </div>
  )
}

export default App
