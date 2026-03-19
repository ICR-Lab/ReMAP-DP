export default function YouTubeDemo() {
  return (
    <section className="py-12">
      <div className="max-w-3xl mx-auto px-6">
        <div className="rounded-2xl overflow-hidden shadow-2xl border border-neutral-200">
          <div className="aspect-video">
            <iframe
              className="w-full h-full"
              src="https://www.youtube.com/embed/9sWFMfBkTWQ"
              title="ReMAP-DP Demo Video"
              allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        </div>
      </div>
    </section>
  )
}
